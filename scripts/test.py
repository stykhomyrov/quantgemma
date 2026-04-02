#!/usr/bin/env python3
"""Evaluate directional accuracy of the fine-tuned model on the test split.

For each bar in each test sequence the model sees all preceding tokens and
predicts the next Z token (volatility-normalized return).  We compare the
predicted direction (up / down) against the actual direction.

  bins  0-11  -> negative return  (down)
  bins 12-23  -> positive return  (up)

A random model scores ~50 %.  Any edge above 50 % is meaningful.

Also reports accuracy by bar position within the sequence to show how
context length influences prediction quality.

Usage:
  uv run scripts/test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

# ---------------------------------------------------------------------------
# Parameters - must match prepare.py and train.py
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data/prepared/v4")
MODEL_PATH = Path("models/checkpoints/v1_bs8_lr3e-05_steps8500")
BASE_PATH  = Path("../quantgemma-research/models/gemma-3-270m")  # for tokenizer

Z_OFF  = 15    # first Z slot index
K_Z    = 24    # number of Z bins
Z_MID  = K_Z // 2   # bins >= Z_MID -> up, < Z_MID -> down

SEQ_BARS = 128  # bars per sequence
BATCH_READ  = 50_000  # parquet read batch size
INFER_BATCH = 32      # sequences per GPU forward pass


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    print(f"checkpoint: {MODEL_PATH}")

    # Build slot -> token ID mapping
    tok = AutoTokenizer.from_pretrained(str(BASE_PATH), local_files_only=True)
    slot_to_id = np.array(
        [tok.convert_tokens_to_ids(f"<unused{i}>") for i in range(73)], dtype=np.int32
    )

    # Z token IDs and a fast reverse lookup
    z_token_ids = slot_to_id[Z_OFF : Z_OFF + K_Z]  # shape (K_Z,)
    id_to_z_bin = {int(tid): b for b, tid in enumerate(z_token_ids)}

    # Load model
    model = Gemma3ForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # Per-bar accuracy tracking
    bar_correct = np.zeros(SEQ_BARS, dtype=np.int64)
    bar_total   = np.zeros(SEQ_BARS, dtype=np.int64)

    z_token_ids_t = torch.tensor(z_token_ids, dtype=torch.long, device=device)
    t0 = time.time()
    n_seqs = 0

    path = DATA_DIR / "split=test" / "part-0.parquet"
    pf = pq.ParquetFile(path)
    seq_len = None  # determined from first batch

    with torch.no_grad():
        for pq_batch in pf.iter_batches(batch_size=BATCH_READ, columns=["token_slots"]):
            col = pq_batch.column("token_slots")
            offsets = col.offsets.to_numpy()
            flat_slots = col.values.to_numpy()

            # Build numpy array of all sequences in this parquet batch
            n = len(col)
            if seq_len is None:
                seq_len = int(offsets[1] - offsets[0])
            ids = np.empty((n, seq_len), dtype=np.int64)
            for j in range(n):
                s, e = int(offsets[j]), int(offsets[j + 1])
                length = min(e - s, seq_len)
                ids[j, :length] = slot_to_id[flat_slots[s:s + length]]

            # Process in GPU batches
            for start in range(0, n, INFER_BATCH):
                end = min(start + INFER_BATCH, n)
                inp = torch.tensor(ids[start:end], dtype=torch.long, device=device)
                all_logits = model(input_ids=inp).logits  # (batch, seq_len, vocab)

                # Score each sequence in the batch
                for k in range(end - start):
                    seq = ids[start + k]
                    logits = all_logits[k]  # (seq_len, vocab)

                    z_bar = 0
                    for pos in range(seq_len - 1):
                        target_id = int(seq[pos + 1])
                        z_bin = id_to_z_bin.get(target_id)
                        if z_bin is None:
                            continue

                        pred_bin = int(logits[pos, z_token_ids_t].argmax().item())
                        hit = int((z_bin >= Z_MID) == (pred_bin >= Z_MID))

                        if z_bar < SEQ_BARS:
                            bar_correct[z_bar] += hit
                            bar_total[z_bar]   += 1
                        z_bar += 1

                n_seqs += end - start
                if n_seqs % (INFER_BATCH * 10) == 0:
                    total   = int(bar_total.sum())
                    correct = int(bar_correct.sum())
                    acc = correct / total if total else 0
                    print(f"  {n_seqs} seqs  acc={acc:.4f}  ({time.time()-t0:.0f}s)")

    total   = int(bar_total.sum())
    correct = int(bar_correct.sum())
    acc = correct / total if total else 0

    print(f"\n{'='*50}")
    print(f"Overall directional accuracy: {acc:.4f}  ({correct}/{total})")
    print(f"Baseline (random):            0.5000")
    print(f"Edge:                         {acc - 0.5:+.4f}")
    print(f"Sequences evaluated:          {n_seqs}")
    print(f"Elapsed:                      {time.time()-t0:.1f}s")

    # Accuracy by bar position (context length effect)
    print(f"\n{'='*50}")
    print(f"Accuracy by bar position (context length effect):")
    print(f"{'Bar':>5}  {'Acc':>7}  {'Correct':>8}  {'Total':>8}")
    print(f"{'-'*5}  {'-'*7}  {'-'*8}  {'-'*8}")
    for b in range(SEQ_BARS):
        if bar_total[b] == 0:
            continue
        ba = bar_correct[b] / bar_total[b]
        print(f"{b:5d}  {ba:7.4f}  {bar_correct[b]:8d}  {bar_total[b]:8d}")

    # Summary by bar ranges
    print(f"\nAccuracy by context depth:")
    ranges = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 63)]
    for lo, hi in ranges:
        hi = min(hi, SEQ_BARS - 1)
        c = int(bar_correct[lo:hi+1].sum())
        t = int(bar_total[lo:hi+1].sum())
        if t == 0:
            continue
        a = c / t
        print(f"  bars {lo:2d}-{hi:2d}:  {a:.4f}  ({c}/{t})")


if __name__ == "__main__":
    main()
