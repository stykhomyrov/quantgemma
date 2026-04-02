# 10m bars continued from 5m checkpoint

**Date:** 2026-04-01
**Status:** Abandoned early

## Setup

- Continued finetuning from v2 checkpoint (v1_bs8_lr3e-05_steps9742, trained on 5m bars)
- 10m bars, SIGMA_WIN=144, all 748 symbols
- Data: v3, 1.84M train / 229K val / 229K test sequences
- LR=3e-5 (same as original training), BATCH_SIZE=8

## Observations

- Loss started at ~2.1 (v2 checkpoint ended at 1.98)
- Quickly settled back to ~2.0 plateau within ~300 steps
- Ran ~2050 steps (~13 minutes) before stopping — no improvement trend
- Same plateau as 5m all-symbol training (v2), suggesting LR=3e-5 is too high for continuation finetuning

## Previous context

- v2 (5m bars, all symbols, 1 hour): val_loss=1.985, directional accuracy ~50% (random)
- v1 (5m bars, BTCUSDT only, 1000 steps): val_loss=2.051

## Takeaway

- Continuing from a checkpoint with the same LR as initial training doesn't help
- The model oscillates around the same loss rather than converging deeper
- Next: try lower LR (5e-6) and training from base model on 10m bars
