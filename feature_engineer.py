#!/usr/bin/env python3
"""
Project Coral Sentinel - Phase 2: Feature Engineering Pipeline
Ingests massive_coral_training_data.csv, engineers time-series context features,
scales the data (leakage-safe), and exports the final artifacts.

Outputs
-------
  engineered_coral_data.csv  — scaled features + unscaled ph column
  advanced_scaler.pkl        — fitted StandardScaler for inference use
"""

import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── config ────────────────────────────────────────────────────────────────────
INPUT_CSV    = "massive_coral_training_data.csv"
OUTPUT_CSV   = "engineered_coral_data.csv"
SCALER_PKL   = "advanced_scaler.pkl"
TARGET_COL   = "ph"

ROLL_SHORT   = 7    # days
ROLL_LONG    = 30   # days  (drives minimum NaN rows to drop)

SORT_KEYS    = ["year", "day_of_year"]   # req 1 — chronological order

# ── 1. ingest ─────────────────────────────────────────────────────────────────
print("=" * 68)
print(" 🪸  CORAL SENTINEL — FEATURE ENGINEERING PIPELINE")
print("=" * 68)

print(f"\n[1/6] Loading '{INPUT_CSV}'...")
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    sys.exit(
        f"ERROR: '{INPUT_CSV}' not found. "
        "Run data_generator.py first to produce the source CSV."
    )

print(f"      Loaded  : {len(df):>10,} rows  ×  {df.shape[1]} columns")

# ── 2. chronological sort (req 1) ─────────────────────────────────────────────
print(f"\n[2/6] Sorting chronologically by {SORT_KEYS}...")
df.sort_values(SORT_KEYS, inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"      First record  →  year={df['year'].iloc[0]}  "
      f"day={df['day_of_year'].iloc[0]}")
print(f"      Last  record  →  year={df['year'].iloc[-1]}  "
      f"day={df['day_of_year'].iloc[-1]}")

# ── 3. vectorised time-series features (req 2) ────────────────────────────────
#   .rolling() and .diff() operate on the already-sorted index.
#   No Python for-loops; every operation is a single vectorised Pandas call.
print(f"\n[3/6] Engineering time-series features (rolling + diff)...")

# 7-day rolling means
df["temp_rolling_7d"]  = df["temperature"].rolling(window=ROLL_SHORT).mean()
df["co2_rolling_7d"]   = df["co2"].rolling(window=ROLL_SHORT).mean()

# 30-day rolling means
df["temp_rolling_30d"] = df["temperature"].rolling(window=ROLL_LONG).mean()
df["co2_rolling_30d"]  = df["co2"].rolling(window=ROLL_LONG).mean()

# 24-hour (1-step) rate of change
df["temp_diff_24h"]    = df["temperature"].diff()
df["co2_diff_24h"]     = df["co2"].diff()

new_cols = [
    "temp_rolling_7d", "co2_rolling_7d",
    "temp_rolling_30d", "co2_rolling_30d",
    "temp_diff_24h",    "co2_diff_24h",
]
print(f"      New columns  : {new_cols}")
nan_before = df.isna().any(axis=1).sum()
print(f"      Rows with ≥1 NaN before dropna : {nan_before:,}")

# ── 4. aggressive NaN removal (req 3) ─────────────────────────────────────────
print(f"\n[4/6] Dropping all rows containing NaN...")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"      Rows after dropna : {len(df):>10,}  "
      f"(dropped {nan_before:,} rows)")

# ── 5. leakage-safe scaling (req 4) ───────────────────────────────────────────
#   Separate the target before fitting the scaler so ph is NEVER seen by
#   StandardScaler — prevents any possibility of target leakage.
print(f"\n[5/6] Scaling features (StandardScaler) — target '{TARGET_COL}' isolated...")

ph_series   = df[TARGET_COL].copy()          # unscaled target, kept aside
feature_df  = df.drop(columns=[TARGET_COL])  # scaler sees ONLY features

scaler      = StandardScaler()
scaled_arr  = scaler.fit_transform(feature_df)

# Rebuild a clean DataFrame with original column names
scaled_df   = pd.DataFrame(scaled_arr, columns=feature_df.columns, index=feature_df.index)

print(f"      Features scaled : {len(feature_df.columns)}")
print(f"      Target isolated : '{TARGET_COL}' (unscaled, will be reattached)")

# Save fitted scaler artifact (req 5)
joblib.dump(scaler, SCALER_PKL)
print(f"      Scaler saved    : '{SCALER_PKL}'")

# ── 6. reattach target & export (req 6) ───────────────────────────────────────
print(f"\n[6/6] Reattaching unscaled '{TARGET_COL}' and exporting CSV...")

# Concatenate so ph is the final column — explicit axis=1 for clarity
final_df = pd.concat([scaled_df, ph_series], axis=1)

final_df.to_csv(OUTPUT_CSV, index=False)
print(f"      Dataset saved   : '{OUTPUT_CSV}'")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  ✅  FEATURE ENGINEERING COMPLETE")
print("=" * 68)
print(f"\n  Final dataset shape : {final_df.shape[0]:,} rows  ×  {final_df.shape[1]} columns")
print(f"  Feature columns     : {final_df.shape[1] - 1}  (scaled)")
print(f"  Target column       : '{TARGET_COL}'  (unscaled)")
print(f"  Rows dropped (NaN)  : {nan_before:,}  "
      f"(≈ {nan_before / (len(final_df) + nan_before) * 100:.1f}% of original)")
print(f"\n  Artifacts")
print(f"    {OUTPUT_CSV}")
print(f"    {SCALER_PKL}")

# Quick sanity checks — will raise AssertionError if any invariant is violated
assert TARGET_COL in final_df.columns,          "ph column missing from output"
assert final_df[TARGET_COL].isna().sum() == 0,  "ph contains NaN"
assert final_df.isna().sum().sum() == 0,         "output DataFrame still contains NaN"
# Verify scaler was not contaminated with ph by confirming feature count matches
assert len(scaler.mean_) == len(feature_df.columns), "scaler feature count mismatch"
print("\n  Sanity checks : all passed ✓")
print()
