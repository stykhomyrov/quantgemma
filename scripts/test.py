#!/usr/bin/env python3
"""Evaluate directional accuracy of the fine-tuned model on the test split.

For each bar in each test sequence the model sees all preceding tokens and
predicts the next Z token (volatility-normalized return).  We compare the
predicted direction (up / down) against the actual direction.

  bins  0-11  -> negative return  (down)
  bins 12-23  -> positive return  (up)

A random model scores ~50 %.  Any edge above 50 % is meaningful.

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

DATA_DIR   = Path("data/prepared/v1")
MODEL_PATH = Path("models/checkpoints/v1_bs8_lr3e-05_steps1000")
BASE_PATH  = Path("../quantgemma-research/models/gemma-3-270m")  # for tokenizer

Z_OFF  = 15    # first Z slot index
K_Z    = 24    # number of Z bins
Z_MID  = K_Z // 2   # bins >= Z_MID -> up, < Z_MID -> down


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    # Build slot -> token ID mapping
    tok = AutoTokenizer.from_pretrained(str(BASE_PATH), local_files_only=True)
    slot_to_id = np.array(
        [tok.convert_tokens_to_ids(f"<unused{i}>") for i in range(73)], dtype=np.int32
    )

    # Z token IDs and a fast reverse lookup
    z_token_ids = slot_to_id[Z_OFF : Z_OFF + K_Z] # shape (K_Z,)
    id_to_z_bin = {int(tid): b for b, tid in enumerate(z_token_ids)}

    # Load test sequences and convert slots → token IDs
    seqs_raw = pq.ParquetFile(DATA_DIR / "split=test" / "part-0.parquet") \
                 .read().column("token_slots").to_pylist()
    seqs = [np.array(slot_to_id[s], dtype=np.int64) for s in seqs_raw]
    print(f"test sequences: {len(seqs)}  seq_len: {seqs[0].shape[0]}")

    # Load model
    model = Gemma3ForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # Evaluate
    correct = total = 0
    z_token_ids_t = torch.tensor(z_token_ids, dtype=torch.long, device=device)
    t0 = time.time()

    with torch.no_grad():
        for i, seq in enumerate(seqs):
            batch  = torch.tensor(seq[None], dtype=torch.long, device=device)
            logits = model(input_ids=batch).logits[0]   # (seq_len, vocab)

            # logits[pos] predicts token at pos+1
            targets = seq[1:]
            for pos, target_id in enumerate(targets):
                z_bin = id_to_z_bin.get(int(target_id))
                if z_bin is None:
                    continue

                # predicted bin = argmax over Z token logits only
                z_logits  = logits[pos, z_token_ids_t]
                pred_bin  = int(z_logits.argmax().item())

                actual_up = z_bin  >= Z_MID
                pred_up   = pred_bin >= Z_MID
                correct  += int(actual_up == pred_up)
                total    += 1

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(seqs)}  acc={correct/total:.4f}  ({time.time()-t0:.0f}s)")

    acc = correct / total
    print(f"\ndirectional_accuracy: {acc:.4f}  ({correct}/{total})")
    print(f"baseline (random):    0.5000")
    print(f"edge:                 {acc - 0.5:+.4f}")
    print(f"elapsed:              {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
