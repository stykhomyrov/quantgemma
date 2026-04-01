#!/usr/bin/env python3
"""Fine-tune Gemma 3 270M on prepared BTCUSDT token sequences.

Loads data from data/prepared/v1, maps token slots to Gemma <unusedN> IDs,
and fine-tunes with causal LM loss. Saves checkpoint to models/checkpoints/.

Usage:
  uv run scripts/train.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import os

import mlflow
import numpy as np

os.environ.setdefault("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", "-1")
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

DATA_DIR    = Path("data/prepared/v1")
MODEL_PATH  = Path("../quantgemma-research/models/gemma-3-270m")
CKPT_DIR    = Path("models/checkpoints")

MLFLOW_DB         = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "quantgemma"
MLFLOW_ARTIFACTS  = "gs://quantgemma/mlflow-artifacts"

TIME_BUDGET  = 3600       # seconds
MAX_STEPS    = 1000       # hard step limit (set to None to rely on TIME_BUDGET only)
BATCH_SIZE   = 8
LR           = 3e-5
WEIGHT_DECAY = 0.05
WARMUP_FRAC  = 0.05
MAX_NORM     = 1.0
SEED         = 7
LOG_EVERY    = 25
EVAL_BATCHES = 32         # batches for validation loss estimate


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(split: str, slot_to_id: np.ndarray) -> np.ndarray:
    """Load token_slots for a split, map to model token IDs, pad to max length."""
    f = pq.ParquetFile(DATA_DIR / f"split={split}" / "part-0.parquet")
    seqs = f.read().column("token_slots").to_pylist()
    max_len = max(len(s) for s in seqs)
    pad_id  = int(slot_to_id[0])   # use slot-0 token as pad (arbitrary, masked in loss)
    out = np.full((len(seqs), max_len), pad_id, dtype=np.int32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = slot_to_id[s]
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_DB)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    _setup_mlflow()

    # Build slot -> token ID mapping from tokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    slot_to_id = np.array(
        [tok.convert_tokens_to_ids(f"<unused{i}>") for i in range(73)], dtype=np.int32
    )
    print(f"slot_to_id sample: slot0={slot_to_id[0]} slot15={slot_to_id[15]} slot39={slot_to_id[39]}")

    # Load data
    train_ids = load_split("train", slot_to_id)
    val_ids   = load_split("val",   slot_to_id)
    seq_len   = train_ids.shape[1]
    print(f"train={len(train_ids)} val={len(val_ids)} seq_len={seq_len}")

    # Model — full fine-tune
    model = Gemma3ForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY,
    )

    # Calibrate steps in time budget with a short probe
    rng = np.random.default_rng(SEED)
    t_probe = time.time()
    for _ in range(10):
        idx   = rng.integers(0, len(train_ids), BATCH_SIZE)
        batch = torch.tensor(train_ids[idx], dtype=torch.long, device=device)
        model(input_ids=batch, labels=batch).loss.backward()
        optimizer.zero_grad(set_to_none=True)
    probe_sec = time.time() - t_probe
    steps_per_sec = 10 / probe_sec
    total_steps   = max(100, int((TIME_BUDGET - probe_sec) * steps_per_sec) + 10)
    warmup_steps  = max(1, int(total_steps * WARMUP_FRAC))
    print(f"probe: {steps_per_sec:.2f} steps/s → {total_steps} total steps, {warmup_steps} warmup")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_bs{BATCH_SIZE}_lr{LR:.0e}_wd{WEIGHT_DECAY}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "weight_decay":  WEIGHT_DECAY,
            "warmup_frac":   WARMUP_FRAC,
            "max_norm":      MAX_NORM,
            "seed":          SEED,
            "time_budget":   TIME_BUDGET,
            "total_steps":   total_steps,
            "seq_len":       seq_len,
            "data_dir":      str(DATA_DIR),
        })

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training loop
        t_start = time.time() - probe_sec
        step = 0
        model.train()

        while time.time() - t_start < TIME_BUDGET and (MAX_STEPS is None or step < MAX_STEPS):
            idx   = rng.integers(0, len(train_ids), BATCH_SIZE)
            batch = torch.tensor(train_ids[idx], dtype=torch.long, device=device)
            loss  = model(input_ids=batch, labels=batch).loss

            if torch.isnan(loss) or loss.item() > 100:
                print(f"ABORT: loss={loss.item():.4f} at step {step}")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t_start
                current_lr = scheduler.get_last_lr()[0] * LR
                print(f"  step {step:5d} | loss={loss.item():.4f} | lr={current_lr:.2e} | {elapsed:.0f}s")
                mlflow.log_metric("train_loss", loss.item(), step=step)
                mlflow.log_metric("lr", current_lr, step=step)

        training_seconds = time.time() - t_start

        # Evaluation
        @torch.no_grad()
        def eval_loss(ids: np.ndarray) -> float:
            model.eval()
            total = 0.0
            for i in range(EVAL_BATCHES):
                idx   = rng.integers(0, len(ids), BATCH_SIZE)
                batch = torch.tensor(ids[idx], dtype=torch.long, device=device)
                total += model(input_ids=batch, labels=batch).loss.item()
            model.train()
            return total / EVAL_BATCHES

        val_loss   = eval_loss(val_ids)
        train_loss = eval_loss(train_ids)
        peak_vram  = torch.cuda.max_memory_allocated() / 1024 ** 2

        mlflow.log_metrics({
            "val_loss":         val_loss,
            "train_loss_final": train_loss,
            "peak_vram_mb":     peak_vram,
            "training_seconds": training_seconds,
            "steps_completed":  step,
        })

        # Save checkpoint locally
        ckpt_name = f"v1_bs{BATCH_SIZE}_lr{LR:.0e}_steps{step}"
        ckpt_path = CKPT_DIR / ckpt_name
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_path))
        print(f"checkpoint:        {ckpt_path}")

        # Upload to GCS only if this is the best val_loss so far
        finished_runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT],
            filter_string="status = 'FINISHED'",
        )
        best_val_loss = finished_runs["metrics.val_loss"].min() if not finished_runs.empty else float("inf")
        if val_loss < best_val_loss:
            print(f"new best val_loss {val_loss:.6f} (prev {best_val_loss:.6f}) — uploading to GCS")
            mlflow.log_artifacts(str(ckpt_path),       artifact_path="checkpoint")
            mlflow.log_artifact("scripts/train.py",    artifact_path="source")
            mlflow.log_artifact("scripts/prepare.py",  artifact_path="source")
        else:
            print(f"val_loss {val_loss:.6f} >= best {best_val_loss:.6f} — skipping GCS upload")

        print(f"val_loss:          {val_loss:.6f}")
        print(f"train_loss:        {train_loss:.6f}")
        print(f"peak_vram_mb:      {peak_vram:.1f}")
        print(f"training_seconds:  {training_seconds:.1f}")
        print(f"total_steps:       {step}")


if __name__ == "__main__":
    train()
