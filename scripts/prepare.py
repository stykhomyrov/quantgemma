#!/usr/bin/env python3
"""Prepare tokenized sequences from raw Binance futures parquet files.

Reads 1-min data for all available symbols, aggregates to 5-min bars,
computes Z and V features, splits 80/10/10 by time per symbol, fits global
bin edges on all train data combined, builds token slot sequences, and writes
per-split parquets + dataset.toml to data/prepared/v2.

Usage:
  uv run scripts/prepare.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR     = Path("data/binance_futures")
OUT_DIR     = Path("data/prepared/v3")

M           = 10    # bar interval (minutes)
SIGMA_WIN   = 144   # rolling std window (m-min bars, ~1 day)
K_Z         = 24    # Z bins
K_V         = 24    # V bins
K_TOD       = 8     # time-of-day slots

SEQ_BARS    = 64    # bars per sequence
STRIDE_BARS = 32    # stride between sequences

TRAIN_FRAC  = 0.8
VAL_FRAC    = 0.1   # test = remaining 0.1

# Token slot offsets: DoW[0-6], ToD[7-14], Z[15-38], V[39-62]  (63 total <= 73)
DOW_OFF = 0
TOD_OFF = 7
Z_OFF   = 7 + K_TOD
V_OFF   = 7 + K_TOD + K_Z

# Minimum number of sequences required to include a symbol
MIN_TRAIN_SEQS = 10


def load_1m_bars(symbol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all 1-min parquet files for a symbol.

    Returns (open_times_ms, closes, quote_volumes) sorted by time, deduped.
    """
    files = sorted((RAW_DIR / symbol).glob(f"{symbol}-1m-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {RAW_DIR / symbol}")

    chunks_t, chunks_c, chunks_v = [], [], []
    for f in files:
        tbl = pq.read_table(f, columns=["open_time", "close", "quote_volume"])
        chunks_t.append(tbl["open_time"].cast(pa.timestamp("ms")).cast(pa.int64()).to_pylist())
        chunks_c.append(tbl["close"].to_pylist())
        chunks_v.append(tbl["quote_volume"].to_pylist())

    t = np.asarray(sum(chunks_t, []), dtype=np.int64)
    c = np.asarray(sum(chunks_c, []), dtype=np.float64)
    v = np.asarray(sum(chunks_v, []), dtype=np.float64)

    order = np.argsort(t, kind="stable")
    t, c, v = t[order], c[order], v[order]
    _, idx = np.unique(t, return_index=True)
    return t[idx], c[idx], v[idx]


def aggregate_bars(
    t1m: np.ndarray, c1m: np.ndarray, qv1m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate 1-min bars into non-overlapping M-min bars."""
    n = (len(c1m) // M) * M
    t = t1m[:n].reshape(-1, M)
    c = c1m[:n].reshape(-1, M)
    v = qv1m[:n].reshape(-1, M)
    return t[:, -1], c[:, -1], v.sum(axis=1)


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation via cumulative sums (O(n))."""
    n = len(values)
    out = np.full(n, np.nan)
    if window <= 1 or n < window:
        return out
    cs  = np.cumsum(values)
    cs2 = np.cumsum(values ** 2)
    ws      = np.empty(n - window + 1)
    ws[0]   = cs[window - 1]
    ws[1:]  = cs[window:] - cs[:n - window]
    ws2     = np.empty(n - window + 1)
    ws2[0]  = cs2[window - 1]
    ws2[1:] = cs2[window:] - cs2[:n - window]
    var = ws2 / window - (ws / window) ** 2
    np.maximum(var, 0.0, out=var)
    out[window - 1:] = np.sqrt(var)
    return out


def compute_features(
    bar_ts: np.ndarray, bar_close: np.ndarray, bar_vol: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute Z, V, ToD, DoW features; drop warmup rows with NaN."""
    ret   = np.log(bar_close[1:] / bar_close[:-1])
    sigma = rolling_std(ret, window=SIGMA_WIN)
    z     = ret / sigma

    with np.errstate(divide="ignore", invalid="ignore"):
        dv = np.log(bar_vol[1:] / bar_vol[:-1])

    ts   = bar_ts[1:]
    secs = ts // 1000
    tod  = ((secs % 86400) // 3600 // (24 // K_TOD)).astype(np.int32)
    dow  = ((secs // 86400 - 3) % 7).astype(np.int32)   # 1970-01-01 was Thursday

    mask = np.isfinite(z) & np.isfinite(dv)
    return {"ts": ts[mask], "Z": z[mask], "V": dv[mask], "ToD": tod[mask], "DoW": dow[mask]}


def assign_splits(n: int) -> np.ndarray:
    train_end = max(1, min(int(n * TRAIN_FRAC), n - 2))
    val_end   = max(train_end + 1, min(int(n * (TRAIN_FRAC + VAL_FRAC)), n - 1))
    labels    = np.empty(n, dtype="U5")
    labels[:train_end]        = "train"
    labels[train_end:val_end] = "val"
    labels[val_end:]          = "test"
    return labels


def quantile_edges(values: np.ndarray, k: int) -> np.ndarray:
    q = np.linspace(1.0 / k, (k - 1) / k, k - 1)
    return np.quantile(values, q)


def to_bins(values: np.ndarray, edges: np.ndarray, k: int) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right"), 0, k - 1).astype(np.int32)


def build_sequences(
    ts: np.ndarray,
    z_bins: np.ndarray, v_bins: np.ndarray,
    tod_bins: np.ndarray, dow_bins: np.ndarray,
    split: str,
) -> dict[str, list]:
    """Build token slot sequences: DoW at day boundaries, ToD as bar delimiter."""
    n = len(ts)
    rows: dict[str, list] = {"split": [], "start_ts": [], "end_ts": [], "token_slots": []}
    if n < SEQ_BARS:
        return rows

    for start in range(0, n - SEQ_BARS + 1, STRIDE_BARS):
        end = start + SEQ_BARS
        ids: list[int] = []
        prev_dow = -1
        for i in range(start, end):
            d = int(dow_bins[i])
            if d != prev_dow:
                ids.append(DOW_OFF + d)
                prev_dow = d
            ids.append(TOD_OFF + int(tod_bins[i]))
            ids.append(Z_OFF   + int(z_bins[i]))
            ids.append(V_OFF   + int(v_bins[i]))
        rows["split"].append(split)
        rows["start_ts"].append(int(ts[start]))
        rows["end_ts"].append(int(ts[end - 1]))
        rows["token_slots"].append(ids)

    return rows


def load_symbol_features(symbol: str) -> dict | None:
    """Load and compute features for a single symbol. Returns None on failure."""
    try:
        t1m, c1m, qv1m = load_1m_bars(symbol)
        bar_ts, bar_close, bar_vol = aggregate_bars(t1m, c1m, qv1m)
        feats = compute_features(bar_ts, bar_close, bar_vol)
        if len(feats["ts"]) < SEQ_BARS * 2:
            return None
        splits = assign_splits(len(feats["ts"]))
        return {"feats": feats, "splits": splits}
    except Exception:
        return None


def main() -> None:
    t0 = time.time()
    sys.stdout.reconfigure(line_buffering=True)

    symbols = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())
    print(f"Found {len(symbols)} symbols")

    # Pass 1: collect only train Z/V for global bin fitting (one symbol at a time)
    print("Pass 1: fitting bin edges ...")
    all_train_z, all_train_v = [], []
    valid_symbols: list[str] = []

    for symbol in symbols:
        data = load_symbol_features(symbol)
        if data is None:
            continue
        train_mask = data["splits"] == "train"
        if train_mask.sum() < SEQ_BARS:
            continue
        valid_symbols.append(symbol)
        all_train_z.append(data["feats"]["Z"][train_mask])
        all_train_v.append(data["feats"]["V"][train_mask])

    print(f"  {len(valid_symbols)} valid symbols ({time.time() - t0:.0f}s)")

    z_edges = quantile_edges(np.concatenate(all_train_z), K_Z)
    v_edges = quantile_edges(np.concatenate(all_train_v), K_V)
    n_train_bars = sum(len(z) for z in all_train_z)
    print(f"  bin edges fit on {n_train_bars:,} train bars")

    # Free pass-1 memory
    del all_train_z, all_train_v

    # Pass 2: re-load each symbol, build sequences, write incrementally to parquet
    print("Pass 2: building sequences ...")
    writers: dict[str, pq.ParquetWriter] = {}
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for i, symbol in enumerate(valid_symbols):
        data = load_symbol_features(symbol)
        if data is None:
            continue
        feats  = data["feats"]
        splits = data["splits"]
        z_b    = to_bins(feats["Z"],   z_edges, K_Z)
        v_b    = to_bins(feats["V"],   v_edges, K_V)
        tod_b  = np.clip(feats["ToD"], 0, K_TOD - 1)
        dow_b  = np.clip(feats["DoW"], 0, 6)

        for split_name in ("train", "val", "test"):
            mask = splits == split_name
            rows = build_sequences(
                ts=feats["ts"][mask], z_bins=z_b[mask], v_bins=v_b[mask],
                tod_bins=tod_b[mask], dow_bins=dow_b[mask], split=split_name,
            )
            if not rows["token_slots"]:
                continue
            table = pa.Table.from_pydict(rows)
            if split_name not in writers:
                path = OUT_DIR / f"split={split_name}" / "part-0.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                writers[split_name] = pq.ParquetWriter(path, table.schema, compression="snappy")
            writers[split_name].write_table(table)
            counts[split_name] += len(rows["token_slots"])

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(valid_symbols)} symbols processed")

    for w in writers.values():
        w.close()

    print(f"  train={counts['train']}  val={counts['val']}  test={counts['test']}")
    print(f"  output={OUT_DIR}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
