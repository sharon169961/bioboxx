#!/usr/bin/env python3
"""
Project Coral Sentinel - Phase 3: Model Training & Hyperparameter Optimization
Trains an XGBoost Regressor on engineered_coral_data.csv with memory-safe
histogram-based tree construction and randomized hyperparameter search.

Outputs
-------
  advanced_coral_model.json  — best XGBoost model (native XGBoost format)
  model_metadata.json        — best params + evaluation metrics
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
INPUT_CSV    = "engineered_coral_data.csv"
MODEL_OUT    = "advanced_coral_model.json"
METADATA_OUT = "model_metadata.json"
TARGET_COL   = "ph"

TEST_SIZE    = 0.20
RANDOM_STATE = 42
CV_FOLDS     = 3
N_ITER       = 15

# ── helpers ───────────────────────────────────────────────────────────────────
def _section(title: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")

def _elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTION (req 1)
# ═════════════════════════════════════════════════════════════════════════════
_section("STEP 1/5 — Data Ingestion")

# Req 1: pd.read_csv wrapped in try/except
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    sys.exit(
        f"ERROR: '{INPUT_CSV}' not found.\n"
        "Run feature_engineer.py first to produce the source CSV."
    )
except Exception as exc:
    sys.exit(f"ERROR: Failed to read '{INPUT_CSV}': {exc}")

print(f"  Loaded       : {len(df):>10,} rows  ×  {df.shape[1]} columns")

# Validate target exists
if TARGET_COL not in df.columns:
    sys.exit(f"ERROR: Target column '{TARGET_COL}' not found in {INPUT_CSV}.")

# Req 1: separate target from features — ph is the ONLY unscaled column
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f"  Feature cols : {X.shape[1]}")
print(f"  Target col   : '{TARGET_COL}'  (unscaled pH)")
print(f"  y range      : [{y.min():.4f}, {y.max():.4f}]")

# ═════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT (req 1)
# ═════════════════════════════════════════════════════════════════════════════
_section("STEP 2/5 — Train / Test Split (80 / 20)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,          # shuffle before split; data is already time-sorted
)

print(f"  Train set    : {X_train.shape[0]:>10,} rows")
print(f"  Test  set    : {X_test.shape[0]:>10,} rows")

# ═════════════════════════════════════════════════════════════════════════════
# 3. MODEL + HYPERPARAMETER SEARCH (reqs 2, 3, 4)
# ═════════════════════════════════════════════════════════════════════════════
_section("STEP 3/5 — RandomizedSearchCV Hyperparameter Optimization")

# Req 4: exact parameter distributions specified
param_distributions = {
    "max_depth"     : [5, 7, 9],
    "learning_rate" : [0.01, 0.05, 0.1],
    "n_estimators"  : [200, 500],
    "subsample"     : [0.8, 1.0],
}

# Req 2: tree_method='hist' for memory efficiency; n_jobs=-1 for all cores
base_model = xgb.XGBRegressor(
    tree_method  = "hist",    # histogram-based splits: O(n) vs O(n log n) exact
    n_jobs       = -1,        # parallelise across all available CPU cores
    random_state = RANDOM_STATE,
    verbosity    = 0,         # suppress XGBoost's own training output
)

# Req 3: RandomizedSearchCV, NOT GridSearchCV; n_iter=15, cv=3, reproducible
search = RandomizedSearchCV(
    estimator          = base_model,
    param_distributions = param_distributions,
    n_iter             = N_ITER,
    cv                 = CV_FOLDS,
    scoring            = "neg_root_mean_squared_error",
    n_jobs             = -1,        # parallelise CV folds across all cores
    random_state       = RANDOM_STATE,
    refit              = True,      # refit best estimator on full train set
    verbose            = 2,
)

print(f"\n  Base model   : XGBRegressor(tree_method='hist', n_jobs=-1)")
print(f"  Search space : {param_distributions}")
print(f"  Strategy     : RandomizedSearchCV  n_iter={N_ITER}  cv={CV_FOLDS}")
print(f"  Scoring      : neg_root_mean_squared_error")
print(f"\n  Starting search on {X_train.shape[0]:,} training rows...")

t0 = time.time()
search.fit(X_train, y_train)
search_time = _elapsed(t0)

print(f"\n  Search complete in {search_time}")
print(f"  Best CV RMSE : {-search.best_score_:.6f}")
print(f"  Best params  : {search.best_params_}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION ON HELD-OUT TEST SET (req 5)
# ═════════════════════════════════════════════════════════════════════════════
_section("STEP 4/5 — Test-Set Evaluation")

best_model = search.best_estimator_
y_pred     = best_model.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"\n  {'Metric':<6}   {'Value':>12}   {'Meaning'}")
print(f"  {'-'*58}")
print(f"  {'R²':<6}   {r2:>12.6f}   "
      f"{'(1.0 = perfect; >0.95 = excellent)':}")
print(f"  {'RMSE':<6}   {rmse:>12.6f}   pH units (lower = better)")
print(f"  {'MAE':<6}   {mae:>12.6f}   pH units (lower = better)")

# Feature importance top-10
importances = pd.Series(
    best_model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print(f"\n  Top-10 feature importances:")
for rank, (feat, imp) in enumerate(importances.head(10).items(), 1):
    bar = "█" * int(imp * 400)
    print(f"    {rank:>2}. {feat:<30}  {imp:.4f}  {bar}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. ARTIFACT EXPORT (req 6)
# ═════════════════════════════════════════════════════════════════════════════
_section("STEP 5/5 — Artifact Export")

# Req 6a: save model in native XGBoost JSON format (portable, version-safe)
best_model.save_model(MODEL_OUT)
print(f"  Model saved  : '{MODEL_OUT}'  (XGBoost native JSON)")

# Req 6b: extract best_params_ + metrics → model_metadata.json
metadata = {
    "project"         : "Coral Sentinel",
    "phase"           : "3 — Model Training",
    "training_rows"   : int(X_train.shape[0]),
    "test_rows"       : int(X_test.shape[0]),
    "feature_count"   : int(X.shape[1]),
    "feature_names"   : list(X.columns),
    "target_column"   : TARGET_COL,
    "search_strategy" : {
        "method"       : "RandomizedSearchCV",
        "n_iter"       : N_ITER,
        "cv_folds"     : CV_FOLDS,
        "scoring"      : "neg_root_mean_squared_error",
        "random_state" : RANDOM_STATE,
    },
    "best_params"     : search.best_params_,
    "cv_best_rmse"    : float(-search.best_score_),
    "test_metrics"    : {
        "r2_score"     : float(r2),
        "rmse"         : float(rmse),
        "mae"          : float(mae),
    },
    "top10_features"  : {k: float(v) for k, v in importances.head(10).items()},
    "search_time"     : search_time,
    "xgboost_version" : xgb.__version__,
}

with open(METADATA_OUT, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Metadata     : '{METADATA_OUT}'  (JSON)")

# ── final summary ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 68}")
print(f"  ✅  TRAINING PIPELINE COMPLETE")
print(f"{'=' * 68}")
print(f"\n  Final dataset : {len(df):,} rows  ×  {X.shape[1]} features")
print(f"  R²  on test   : {r2:.6f}")
print(f"  RMSE on test  : {rmse:.6f} pH units")
print(f"  MAE  on test  : {mae:.6f} pH units")
print(f"\n  Artifacts")
print(f"    {MODEL_OUT}")
print(f"    {METADATA_OUT}")
print()
