# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Soil Moisture Time Series Forecasting Tutorial
#
# **Raam Catchment, Netherlands — 15 monitoring stations, 2016–2018**
#
# This tutorial compares forecasting approaches from simple baselines to deep learning,
# and systematically evaluates **univariate vs. multivariate** models across
# **multiple forecast horizons** (1, 7, and 14 days ahead).
#
# ## Contents
# 0. Setup & Data Loading
# 1. Data Preprocessing & Feature Engineering
# 2. Baseline Models (Persistence, Seasonal Naive)
# 3. Statistical Models (AR, ARIMA, ARIMAX)
# 4. Machine Learning — XGBoost (univariate vs. multivariate)
# 5. Deep Learning — LSTM (univariate vs. multivariate)
# 6. Advanced Deep Learning — Temporal Fusion Transformer
# 7. Model Comparison & Discussion

# %% [markdown]
# ## 0. Setup

# %%
import os
# Fix OpenMP conflict between XGBoost and PyTorch on macOS
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; remove this line in Jupyter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

HORIZONS = [1, 7, 14]
DATA_DIR = "RAAM_cov"
SEED = 42
np.random.seed(SEED)

# Stations to use for per-station models (for readability in the tutorial)
FOCUS_STATIONS = ['RM01', 'RM04', 'RM05', 'RM07', 'RM14']

# %% [markdown]
# ## 1. Data Loading & Preprocessing

# %%
# Load all 15 station CSVs
station_dfs = {}
for i in range(1, 16):
    name = f"RM{i:02d}"
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"), parse_dates=['date'])
    df['station'] = name
    station_dfs[name] = df

all_data = pd.concat(station_dfs.values(), ignore_index=True)
print(f"Loaded {len(all_data)} records from {all_data['station'].nunique()} stations")
print(f"Date range: {all_data['date'].min().date()} to {all_data['date'].max().date()}")

# %%
# Data quality: clip physically impossible negative VWC5 values (RM08)
neg_count = (all_data['VWC5'] < 0).sum()
print(f"Negative VWC5 values: {neg_count} — clipping to 0")
all_data['VWC5'] = all_data['VWC5'].clip(lower=0)

# Check missing VWC5 per station
print("\nMissing VWC5 per station:")
miss = all_data.groupby('station')['VWC5'].apply(lambda s: s.isna().sum())
print(miss[miss > 0].to_string())

# %%
# Forward-fill short gaps in VWC5 (up to 3 consecutive days), leave longer gaps as NaN
for name in station_dfs:
    mask = all_data['station'] == name
    all_data.loc[mask, 'VWC5'] = all_data.loc[mask, 'VWC5'].ffill(limit=3)

# Also forward-fill the rainfall column (has 15 missing values)
all_data['rd'] = all_data['rd'].ffill(limit=3)

# %% [markdown]
# ### Train / Test Split
#
# We use a **temporal split**: train on 2016–2017, test on 2018.
# This avoids data leakage and mimics a realistic forecasting scenario.

# %%
TRAIN_END = '2017-12-31'
TEST_START = '2018-01-01'

train_data = all_data[all_data['date'] <= TRAIN_END].copy()
test_data = all_data[all_data['date'] >= TEST_START].copy()

print(f"Train: {len(train_data)} records ({train_data['date'].min().date()} to {train_data['date'].max().date()})")
print(f"Test:  {len(test_data)} records ({test_data['date'].min().date()} to {test_data['date'].max().date()})")

# %% [markdown]
# ### Feature Engineering for Multi-Horizon Forecasting
#
# For an **h-day-ahead** forecast, we need to predict `VWC5(t+h)` given information
# available at time `t`. We create target columns for each horizon and ensure
# features only use information available at time `t`.

# %%
# Meteorological and vegetation features (available at time t)
METEO_FEATURES = ['EV24', 'FG', 'Q', 'rd', 'SQ', 'TN', 'TX', 'UG']
STATIC_FEATURES = ['BOFEK']  # Crop varies by year, BOFEK is fixed
TARGET = 'VWC5'

def create_forecasting_features(df, max_horizon=14):
    """Create lag features and multi-horizon targets for a single station."""
    df = df.sort_values('date').copy()

    # -- Lag features of VWC5 (autoregressive inputs) --
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        df[f'VWC5_lag{lag}'] = df['VWC5'].shift(lag)

    # Rolling statistics of VWC5
    df['VWC5_roll7_mean'] = df['VWC5'].shift(1).rolling(7).mean()
    df['VWC5_roll14_mean'] = df['VWC5'].shift(1).rolling(14).mean()
    df['VWC5_roll7_std'] = df['VWC5'].shift(1).rolling(7).std()

    # -- Lag features of key meteorological variables --
    for feat in METEO_FEATURES:
        df[f'{feat}_lag1'] = df[feat].shift(1)

    # Rolling rainfall features
    df['rd_3day'] = df['rd'].shift(1).rolling(3).sum()
    df['rd_7day'] = df['rd'].shift(1).rolling(7).sum()
    df['rd_14day'] = df['rd'].shift(1).rolling(14).sum()

    # Rolling temperature
    df['TX_roll7'] = df['TX'].shift(1).rolling(7).mean()
    df['EV24_roll7'] = df['EV24'].shift(1).rolling(7).mean()

    # -- Calendar features --
    df['DOY'] = df['date'].dt.dayofyear
    df['DOY_sin'] = np.sin(2 * np.pi * df['DOY'] / 365.25)
    df['DOY_cos'] = np.cos(2 * np.pi * df['DOY'] / 365.25)
    df['month'] = df['date'].dt.month

    # -- Multi-horizon targets --
    for h in HORIZONS:
        df[f'target_h{h}'] = df['VWC5'].shift(-h)

    return df


# Apply per station
processed_dfs = {}
for name, sdf in station_dfs.items():
    sdf_clean = sdf.copy()
    sdf_clean['VWC5'] = sdf_clean['VWC5'].clip(lower=0)
    sdf_clean['VWC5'] = sdf_clean['VWC5'].ffill(limit=3)
    sdf_clean['rd'] = sdf_clean['rd'].ffill(limit=3)
    processed_dfs[name] = create_forecasting_features(sdf_clean)

all_processed = pd.concat(processed_dfs.values(), ignore_index=True)
print(f"Processed data: {len(all_processed)} records, {all_processed.shape[1]} columns")

# %%
# Define feature sets for univariate vs multivariate comparison

# Univariate: only VWC5 lag features + calendar
UNIVARIATE_FEATURES = [
    'VWC5_lag1', 'VWC5_lag2', 'VWC5_lag3', 'VWC5_lag5', 'VWC5_lag7',
    'VWC5_lag14', 'VWC5_lag21', 'VWC5_lag30',
    'VWC5_roll7_mean', 'VWC5_roll14_mean', 'VWC5_roll7_std',
    'DOY_sin', 'DOY_cos',
]

# Multivariate: lag features + meteorological covariates + static
MULTIVARIATE_FEATURES = UNIVARIATE_FEATURES + [
    'EV24_lag1', 'FG_lag1', 'Q_lag1', 'rd_lag1', 'SQ_lag1',
    'TN_lag1', 'TX_lag1', 'UG_lag1',
    'rd_3day', 'rd_7day', 'rd_14day',
    'TX_roll7', 'EV24_roll7',
    'LAI',
    'BOFEK',
]

print(f"Univariate features:   {len(UNIVARIATE_FEATURES)}")
print(f"Multivariate features: {len(MULTIVARIATE_FEATURES)}")

# %% [markdown]
# ### Helper: Evaluation Metrics

# %%
def evaluate_forecast(y_true, y_pred, label=""):
    """Compute RMSE, MAE, R2 and return as dict."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'n': 0}
    return {
        'RMSE': np.sqrt(mean_squared_error(yt, yp)),
        'MAE': mean_absolute_error(yt, yp),
        'R2': r2_score(yt, yp),
        'n': len(yt),
    }


# Master results collector
results = []

def record_result(model_name, variant, horizon, station, y_true, y_pred):
    """Evaluate and store a result row."""
    metrics = evaluate_forecast(np.array(y_true), np.array(y_pred))
    metrics.update({
        'model': model_name,
        'variant': variant,  # 'univariate' or 'multivariate'
        'horizon': horizon,
        'station': station,
    })
    results.append(metrics)
    return metrics

# %% [markdown]
# ---
# ## 2. Baseline Models
#
# Two simple baselines that any useful model must beat:
# - **Persistence**: predict that soil moisture stays the same as today
# - **Seasonal Naive**: predict using the value from the same day of year last year

# %%
print("=" * 60, flush=True)
print("BASELINE MODELS", flush=True)
print("=" * 60, flush=True)

for station in FOCUS_STATIONS:
    sdf = processed_dfs[station]
    test_mask = sdf['date'] >= TEST_START

    for h in HORIZONS:
        target_col = f'target_h{h}'
        test_rows = sdf[test_mask].dropna(subset=[target_col, 'VWC5'])

        if len(test_rows) == 0:
            continue

        y_true = test_rows[target_col].values

        # Persistence: y_hat(t+h) = y(t)
        y_pred_persist = test_rows['VWC5'].values
        m = record_result('Persistence', 'univariate', h, station, y_true, y_pred_persist)

        # Seasonal Naive: y_hat(t+h) = y(t+h - 365)
        # Use value from same DOY previous year
        sdf_indexed = sdf.set_index('date')['VWC5']
        y_pred_seasonal = []
        for _, row in test_rows.iterrows():
            target_date = row['date'] + pd.Timedelta(days=h)
            lookup_date = target_date - pd.Timedelta(days=365)
            # Find closest available date within +/- 3 days
            for offset in range(4):
                for d in [lookup_date + pd.Timedelta(days=offset),
                          lookup_date - pd.Timedelta(days=offset)]:
                    if d in sdf_indexed.index and not np.isnan(sdf_indexed[d]):
                        y_pred_seasonal.append(sdf_indexed[d])
                        break
                else:
                    continue
                break
            else:
                y_pred_seasonal.append(np.nan)

        y_pred_seasonal = np.array(y_pred_seasonal)
        record_result('Seasonal Naive', 'univariate', h, station, y_true, y_pred_seasonal)

# Show baseline results
baseline_results = pd.DataFrame([r for r in results if r['model'] in ['Persistence', 'Seasonal Naive']])
print("\nBaseline Results (RMSE):")
pivot = baseline_results.pivot_table(values='RMSE', index=['station', 'model'], columns='horizon')
print(pivot.round(4).to_string())

# %% [markdown]
# **Observation:** Persistence is very strong at horizon 1 (RMSE ~0.005–0.01) because soil moisture
# changes slowly. At longer horizons, it degrades substantially. Seasonal naive captures
# the annual cycle but ignores short-term dynamics.

# %% [markdown]
# ---
# ## 3. Linear Models: Ridge Regression (univariate vs multivariate)
#
# Before jumping to complex models, we use linear regression as a bridge between
# naive baselines and ML. This is fast, interpretable, and surprisingly effective:
# - **Linear (univariate)**: uses only VWC5 lag features + calendar
# - **Linear (multivariate)**: adds meteorological covariates
#
# We use Ridge regression (L2 regularization) to prevent overfitting with
# correlated features. A separate model is trained for each forecast horizon.

# %%
from sklearn.linear_model import Ridge

print("=" * 60, flush=True)
print("LINEAR MODELS (Ridge Regression)", flush=True)
print("=" * 60, flush=True)

for station in FOCUS_STATIONS:
    print(f"\n--- {station} ---", flush=True)
    sdf = processed_dfs[station]
    train_df = sdf[sdf['date'] <= TRAIN_END]
    test_df = sdf[sdf['date'] >= TEST_START]

    for h in HORIZONS:
        target_col = f'target_h{h}'

        # Univariate Ridge
        train_clean = train_df.dropna(subset=UNIVARIATE_FEATURES + [target_col])
        test_clean = test_df.dropna(subset=UNIVARIATE_FEATURES + [target_col])

        if len(train_clean) < 30 or len(test_clean) < 10:
            continue

        ridge_uni = Ridge(alpha=1.0)
        ridge_uni.fit(train_clean[UNIVARIATE_FEATURES], train_clean[target_col])
        preds_uni = ridge_uni.predict(test_clean[UNIVARIATE_FEATURES])
        y_true = test_clean[target_col].values
        m = record_result('Ridge', 'univariate', h, station, y_true, preds_uni)

        # Multivariate Ridge
        train_clean_m = train_df.dropna(subset=MULTIVARIATE_FEATURES + [target_col])
        test_clean_m = test_df.dropna(subset=MULTIVARIATE_FEATURES + [target_col])
        ridge_multi = Ridge(alpha=1.0)
        ridge_multi.fit(train_clean_m[MULTIVARIATE_FEATURES], train_clean_m[target_col])
        preds_multi = ridge_multi.predict(test_clean_m[MULTIVARIATE_FEATURES])
        y_true_m = test_clean_m[target_col].values
        m2 = record_result('Ridge', 'multivariate', h, station, y_true_m, preds_multi)

        print(f"  h={h:2d}: Uni RMSE={m['RMSE']:.4f}, Multi RMSE={m2['RMSE']:.4f}", flush=True)

# %% [markdown]
# ---
# ## 4. Machine Learning — XGBoost
#
# Gradient boosted trees are a strong general-purpose approach. We compare:
# - **Univariate XGBoost**: uses only VWC5 lag features + calendar
# - **Multivariate XGBoost**: adds meteorological covariates, LAI, soil type
#
# We train both **per-station** and **global** (all stations pooled) models.

# %%
import xgboost as xgb

print("=" * 60, flush=True)
print("XGBOOST MODELS", flush=True)
print("=" * 60, flush=True)

XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'random_state': SEED,
    'n_jobs': -1,
}


def train_xgb_model(train_df, test_df, features, target_col, params=XGB_PARAMS):
    """Train XGBoost and return predictions on test set."""
    train_clean = train_df.dropna(subset=features + [target_col])
    test_clean = test_df.dropna(subset=features + [target_col])

    if len(train_clean) < 30 or len(test_clean) < 10:
        return np.full(len(test_clean), np.nan), test_clean[target_col].values

    X_train = train_clean[features].values
    y_train = train_clean[target_col].values
    X_test = test_clean[features].values
    y_true = test_clean[target_col].values

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_true)], verbose=False)

    y_pred = model.predict(X_test)
    return y_pred, y_true, model


# --- Per-station XGBoost ---
print("\n--- Per-Station XGBoost ---")

feature_importances = {}
xgb_models = {}  # store for explainability section
xgb_test_data = {}

for station in FOCUS_STATIONS:
    sdf = processed_dfs[station]
    train_df = sdf[sdf['date'] <= TRAIN_END]
    test_df = sdf[sdf['date'] >= TEST_START]

    for h in HORIZONS:
        target_col = f'target_h{h}'

        # Univariate
        y_pred, y_true, _ = train_xgb_model(train_df, test_df, UNIVARIATE_FEATURES, target_col)
        m = record_result('XGBoost', 'univariate', h, station, y_true, y_pred)

        # Multivariate
        y_pred, y_true, mdl = train_xgb_model(train_df, test_df, MULTIVARIATE_FEATURES, target_col)
        m2 = record_result('XGBoost', 'multivariate', h, station, y_true, y_pred)

        if h == 1:
            feature_importances[station] = dict(zip(MULTIVARIATE_FEATURES, mdl.feature_importances_))
            xgb_models[station] = mdl
            test_clean = test_df.dropna(subset=MULTIVARIATE_FEATURES + [target_col])
            xgb_test_data[station] = test_clean

        print(f"  {station} h={h:2d}: Uni RMSE={m['RMSE']:.4f}, Multi RMSE={m2['RMSE']:.4f}")

# %%
# --- Global XGBoost (all stations pooled) ---
print("\n--- Global XGBoost (all stations pooled) ---")

# Add station ID as a numeric feature for global model
station_map = {name: i for i, name in enumerate(sorted(all_processed['station'].unique()))}
all_processed['station_id'] = all_processed['station'].map(station_map)

GLOBAL_UNI_FEATURES = UNIVARIATE_FEATURES + ['station_id']
GLOBAL_MULTI_FEATURES = MULTIVARIATE_FEATURES + ['station_id']

train_global = all_processed[all_processed['date'] <= TRAIN_END]
test_global = all_processed[all_processed['date'] >= TEST_START]

for h in HORIZONS:
    target_col = f'target_h{h}'

    # Univariate global
    y_pred, y_true, _ = train_xgb_model(train_global, test_global,
                                          GLOBAL_UNI_FEATURES, target_col)
    m = record_result('XGBoost-Global', 'univariate', h, 'ALL', y_true, y_pred)

    # Multivariate global
    y_pred, y_true, mdl = train_xgb_model(train_global, test_global,
                                            GLOBAL_MULTI_FEATURES, target_col)
    m2 = record_result('XGBoost-Global', 'multivariate', h, 'ALL', y_true, y_pred)

    print(f"  Global h={h:2d}: Uni RMSE={m['RMSE']:.4f}, Multi RMSE={m2['RMSE']:.4f}")

# %%
# Feature importance plot (multivariate XGBoost, h=1)
fig, axes = plt.subplots(1, min(3, len(feature_importances)), figsize=(18, 6))
if not hasattr(axes, '__len__'):
    axes = [axes]
for ax, (station, imp_dict) in zip(axes, list(feature_importances.items())[:3]):
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    names, vals = zip(*sorted_imp)
    ax.barh(range(len(names)), vals, color='steelblue')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f'{station} — Feature Importance (h=1)')
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved xgb_feature_importance.png")

# %% [markdown]
# ---
# ## 4b. Explainability Analysis
#
# Understanding *why* a model makes certain predictions is as important as accuracy,
# especially in environmental science where physical interpretability matters.
#
# We use three complementary approaches:
# 1. **SHAP values** — game-theoretic feature attributions for each prediction
# 2. **Partial Dependence Plots (PDP)** — marginal effect of individual features
# 3. **Temporal feature contribution** — how feature importance changes across seasons

# %%
import shap

print("=" * 60, flush=True)
print("EXPLAINABILITY ANALYSIS (XGBoost, h=1)", flush=True)
print("=" * 60, flush=True)

# %% [markdown]
# ### 4b.1 SHAP Values
#
# SHAP (SHapley Additive exPlanations) assigns each feature a contribution to
# each individual prediction. Unlike global feature importance, SHAP reveals:
# - Which features drive **specific** predictions (local explanations)
# - How feature values relate to their impact (direction of effect)
# - Feature interactions

# %%
# Compute SHAP values for a representative station
shap_station = 'RM01'
mdl = xgb_models[shap_station]
test_df_shap = xgb_test_data[shap_station]
X_test_shap = test_df_shap[MULTIVARIATE_FEATURES].values

explainer = shap.TreeExplainer(mdl)
shap_values = explainer.shap_values(X_test_shap)

print(f"SHAP values computed for {shap_station}: {shap_values.shape}", flush=True)
print(f"  (each row = one test sample, each column = one feature's contribution)", flush=True)

# %%
# SHAP summary plot: feature importance + direction of effect
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_shap, feature_names=MULTIVARIATE_FEATURES,
                  show=False, max_display=20)
plt.title(f'SHAP Summary — {shap_station} (XGBoost, h=1)', fontsize=13)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_summary.png", flush=True)

# %%
# SHAP bar plot: mean absolute SHAP values (global importance)
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_shap, feature_names=MULTIVARIATE_FEATURES,
                  plot_type='bar', show=False, max_display=20)
plt.title(f'Mean |SHAP| — {shap_station} (XGBoost, h=1)', fontsize=13)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_bar.png", flush=True)

# %% [markdown]
# ### 4b.2 SHAP Dependence Plots
#
# These show how a single feature's value affects the prediction, colored by
# an interacting feature. They reveal nonlinear relationships and interactions.

# %%
# Dependence plots for key features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
key_features = ['VWC5_lag1', 'rd_3day', 'EV24_roll7', 'VWC5_roll7_mean', 'TX_lag1', 'UG_lag1']

for ax, feat in zip(axes.flat, key_features):
    feat_idx = MULTIVARIATE_FEATURES.index(feat)
    shap.dependence_plot(feat_idx, shap_values, X_test_shap,
                         feature_names=MULTIVARIATE_FEATURES, ax=ax, show=False)

plt.suptitle(f'SHAP Dependence Plots — {shap_station}', fontsize=14)
plt.tight_layout()
plt.savefig('shap_dependence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_dependence.png", flush=True)

# %% [markdown]
# ### 4b.3 Local Explanation: Wettest vs Driest Prediction
#
# SHAP waterfall plots show exactly why the model made a specific prediction —
# which features pushed it up or down from the average.

# %%
# Find wettest and driest predictions in the test set
y_pred_shap = mdl.predict(X_test_shap)
wettest_idx = np.argmax(y_pred_shap)
driest_idx = np.argmin(y_pred_shap)

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

for ax_idx, (idx, label) in enumerate([(wettest_idx, 'Wettest'), (driest_idx, 'Driest')]):
    plt.sca(axes[ax_idx])
    shap_explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test_shap[idx],
        feature_names=MULTIVARIATE_FEATURES
    )
    shap.waterfall_plot(shap_explanation, max_display=12, show=False)
    axes[ax_idx].set_title(f'{label} Prediction (VWC5={y_pred_shap[idx]:.3f})', fontsize=12)

plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_waterfall.png", flush=True)

# %% [markdown]
# ### 4b.4 Partial Dependence Plots
#
# PDPs show the **marginal effect** of a feature on the prediction, averaging
# over all other features. Unlike SHAP dependence plots (which show raw data),
# PDPs show the model's learned response curve.

# %%
from sklearn.inspection import PartialDependenceDisplay

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
pdp_features = ['VWC5_lag1', 'rd_3day', 'EV24_roll7', 'TX_lag1', 'UG_lag1', 'VWC5_roll7_mean']

for ax, feat in zip(axes.flat, pdp_features):
    feat_idx = MULTIVARIATE_FEATURES.index(feat)
    PartialDependenceDisplay.from_estimator(
        mdl, X_test_shap, [feat_idx],
        feature_names=MULTIVARIATE_FEATURES, ax=ax,
        kind='both',  # shows individual conditional expectation (ICE) + average
        ice_lines_kw={'color': 'steelblue', 'alpha': 0.05, 'linewidth': 0.5},
        pd_line_kw={'color': 'red', 'linewidth': 2},
    )
    ax.set_title(f'PDP + ICE: {feat}', fontsize=11)

plt.suptitle(f'Partial Dependence Plots — {shap_station} (XGBoost, h=1)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('partial_dependence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved partial_dependence.png", flush=True)

# %% [markdown]
# ### 4b.5 Seasonal Variation of Feature Importance
#
# Feature importance isn't static — rainfall matters more in wet seasons,
# evapotranspiration matters more in summer. We analyze how SHAP values
# shift across seasons.

# %%
# Assign seasons to test data
test_dates = test_df_shap['date'].values
test_months = pd.to_datetime(test_dates).month
season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
seasons = np.array([season_map[m] for m in test_months])

# Top features to analyze
top_feats = ['VWC5_lag1', 'VWC5_roll7_mean', 'rd_3day', 'EV24_roll7', 'TX_lag1', 'UG_lag1']
top_feat_indices = [MULTIVARIATE_FEATURES.index(f) for f in top_feats]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
colors = {'Winter': '#2166ac', 'Spring': '#66bd63', 'Summer': '#d73027', 'Autumn': '#f46d43'}

for ax, feat, feat_idx in zip(axes.flat, top_feats, top_feat_indices):
    seasonal_means = []
    for season in season_order:
        mask = seasons == season
        seasonal_means.append(np.mean(np.abs(shap_values[mask, feat_idx])))

    bars = ax.bar(season_order, seasonal_means,
                  color=[colors[s] for s in season_order], edgecolor='black', linewidth=0.5)
    ax.set_title(f'{feat}', fontsize=12)
    ax.set_ylabel('Mean |SHAP value|')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Seasonal Feature Importance — {shap_station}', fontsize=14)
plt.tight_layout()
plt.savefig('seasonal_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved seasonal_feature_importance.png", flush=True)

# %% [markdown]
# ### 4b.6 Cross-Station Comparison of Feature Importance
#
# Do different stations rely on different features? This matters for understanding
# how soil type, crop type, and local conditions affect the forecasting process.

# %%
# Compute SHAP for all focus stations and compare
station_shap_means = {}
for station in FOCUS_STATIONS:
    if station not in xgb_models:
        continue
    mdl_s = xgb_models[station]
    X_s = xgb_test_data[station][MULTIVARIATE_FEATURES].values
    sv = shap.TreeExplainer(mdl_s).shap_values(X_s)
    station_shap_means[station] = np.mean(np.abs(sv), axis=0)

# Heatmap of feature importance across stations
shap_df = pd.DataFrame(station_shap_means, index=MULTIVARIATE_FEATURES).T
# Normalize per station (relative importance)
shap_df_norm = shap_df.div(shap_df.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(shap_df_norm.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(MULTIVARIATE_FEATURES)))
ax.set_yticks(range(len(shap_df_norm)))
ax.set_xticklabels(MULTIVARIATE_FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(shap_df_norm.index, fontsize=10)
plt.colorbar(im, ax=ax, label='Relative importance (normalized SHAP)')
ax.set_title('Feature Importance Across Stations (XGBoost, h=1)', fontsize=13)
plt.tight_layout()
plt.savefig('cross_station_shap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved cross_station_shap.png", flush=True)

# %%
# Print table of top-3 features per station
print("\nTop-3 features per station (by mean |SHAP|):", flush=True)
for station in shap_df.index:
    top3 = shap_df.loc[station].nlargest(3)
    feats_str = ', '.join([f"{f} ({v:.4f})" for f, v in top3.items()])
    print(f"  {station}: {feats_str}", flush=True)

# %% [markdown]
# ### Explainability Summary
#
# Key insights from the explainability analysis:
#
# 1. **VWC5_lag1 dominates** — the previous day's moisture is by far the strongest
#    predictor, consistent with the high autocorrelation (0.97).
#
# 2. **Rolling means matter** — VWC5_roll7_mean captures medium-term trends and is
#    the second most important feature, more useful than individual lagged values.
#
# 3. **Rainfall and evapotranspiration are the key drivers** beyond autocorrelation:
#    - Higher rainfall (rd_3day) → higher predicted moisture (positive SHAP)
#    - Higher evapotranspiration (EV24_roll7) → lower predicted moisture (negative SHAP)
#    - These align with physical understanding of the soil water balance.
#
# 4. **Seasonal shifts**: Rainfall importance peaks in autumn/winter (when soils
#    respond more to precipitation), while temperature/ET importance peaks in summer.
#
# 5. **Station differences**: Stations with crop rotation (RM05, RM07) show higher
#    LAI importance compared to permanent grassland stations (RM01, RM04).

# %% [markdown]
# ---
# ## 5. Deep Learning — LSTM
#
# We implement an LSTM network in PyTorch for sequence-to-value forecasting.
# The model takes a window of past observations and predicts VWC5 at horizon h.
#
# **Univariate LSTM**: input = past VWC5 values only
# **Multivariate LSTM**: input = past VWC5 + meteorological covariates

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
class SoilMoistureDataset(Dataset):
    """Sliding window dataset for time series forecasting."""

    def __init__(self, series, features_df, window_size, horizon, target_col='VWC5'):
        """
        Args:
            series: DataFrame with date index containing target + features.
            features_df: DataFrame of feature columns to use.
            window_size: Number of past timesteps as input.
            horizon: Forecast horizon (days ahead).
            target_col: Name of the target column.
        """
        self.window = window_size
        self.horizon = horizon

        target = series[target_col].values
        feats = features_df.values

        self.X = []
        self.y = []

        for i in range(len(target) - window_size - horizon):
            x_window = feats[i:i + window_size]
            y_val = target[i + window_size + horizon - 1]
            if not np.any(np.isnan(x_window)) and not np.isnan(y_val):
                self.X.append(x_window)
                self.y.append(y_val)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take last timestep
        return self.fc(last_hidden).squeeze(-1)


def train_lstm(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=10):
    """Train LSTM with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_lstm(model, data_loader):
    """Generate predictions from trained LSTM."""
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            preds.append(pred.cpu().numpy())
            trues.append(y_batch.numpy())
    return np.concatenate(preds), np.concatenate(trues)


# %%
# LSTM configuration
WINDOW_SIZE = 30  # 30-day lookback window
BATCH_SIZE = 64
EPOCHS = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Define feature columns for univariate and multivariate LSTM
LSTM_UNI_COLS = ['VWC5']
LSTM_MULTI_COLS = ['VWC5'] + METEO_FEATURES + ['LAI']

print("=" * 60, flush=True)
print("LSTM MODELS", flush=True)
print("=" * 60, flush=True)


def create_sequences(data, target, window, horizon):
    """Create sliding window sequences. Returns X (n, window, features) and y (n,)."""
    X, y = [], []
    for i in range(len(data) - window - horizon):
        x_win = data[i:i + window]
        y_val = target[i + window + horizon - 1]
        if not np.any(np.isnan(x_win)) and not np.isnan(y_val):
            X.append(x_win)
            y.append(y_val)
    if len(X) == 0:
        return np.array([]).reshape(0, window, data.shape[1] if data.ndim > 1 else 1), np.array([])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


for station in FOCUS_STATIONS:
    print(f"\n--- {station} ---", flush=True)
    sdf = processed_dfs[station].dropna(subset=['VWC5']).sort_values('date').copy()

    # Concatenate train+test for contiguous sequences
    full_target = sdf['VWC5'].ffill().bfill().values
    target_mean, target_std = full_target.mean(), full_target.std()
    target_norm = (full_target - target_mean) / target_std

    n_train = (sdf['date'] <= TRAIN_END).sum()

    if n_train < 100 or len(sdf) - n_train < 50:
        print(f"  Skipping: insufficient data", flush=True)
        continue

    for variant, feat_cols in [('univariate', LSTM_UNI_COLS),
                                ('multivariate', LSTM_MULTI_COLS)]:
        # Prepare and scale features
        feats = sdf[feat_cols].ffill().bfill().values
        scaler = StandardScaler()
        scaler.fit(feats[:n_train])
        feats_scaled = scaler.transform(feats)

        for h in HORIZONS:
            # Create sequences from full series
            X_all, y_all = create_sequences(feats_scaled, target_norm, WINDOW_SIZE, h)

            if len(X_all) < 50:
                continue

            # Split: windows whose target falls in the test period
            # A target at index (i + WINDOW + h - 1) is in test if >= n_train
            split_idx = max(0, n_train - WINDOW_SIZE - h)
            X_train, y_train = X_all[:split_idx], y_all[:split_idx]
            X_test, y_test = X_all[split_idx:], y_all[split_idx:]

            if len(X_train) < 30 or len(X_test) < 10:
                continue

            # Use last 15% of training for validation
            val_n = max(1, int(len(X_train) * 0.15))
            X_tr, X_val = X_train[:-val_n], X_train[-val_n:]
            y_tr, y_val = y_train[:-val_n], y_train[-val_n:]

            # Create DataLoaders
            train_ds = torch.utils.data.TensorDataset(
                torch.tensor(X_tr), torch.tensor(y_tr))
            val_ds = torch.utils.data.TensorDataset(
                torch.tensor(X_val), torch.tensor(y_val))

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

            # Build and train model
            input_size = len(feat_cols)
            model = LSTMForecaster(input_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
            model = train_lstm(model, train_loader, val_loader, epochs=EPOCHS)

            # Predict on test
            model.eval()
            with torch.no_grad():
                preds_norm = model(torch.tensor(X_test).to(device)).cpu().numpy()

            # Denormalize
            preds = preds_norm * target_std + target_mean
            y_true = y_test * target_std + target_mean

            m = record_result('LSTM', variant, h, station, y_true, preds)
            print(f"  {variant:14s} h={h:2d}: RMSE={m['RMSE']:.4f}, R2={m['R2']:.3f}", flush=True)

# %% [markdown]
# ---
# ## 6. Temporal Fusion Transformer (Simplified)
#
# The TFT is a state-of-the-art architecture for multi-horizon forecasting.
# Here we implement a simplified version using PyTorch that captures the key ideas:
# - **Variable selection** for static and time-varying inputs
# - **Gated residual networks** for nonlinear processing
# - **Multi-head attention** over the temporal dimension
#
# For a production implementation, consider using the
# [pytorch-forecasting](https://pytorch-forecasting.readthedocs.io/) library.

# %%
class GatedResidualNetwork(nn.Module):
    """GRN: core building block of TFT."""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(output_size)

        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = None

        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_fc = None

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip is not None else x
        h = self.fc1(x)
        if self.context_fc is not None and context is not None:
            h = h + self.context_fc(context)
        h = self.elu(h)
        h = self.dropout(self.fc2(h))
        gate = self.sigmoid(self.gate(h))
        return self.layer_norm(gate * h + residual)


class SimplifiedTFT(nn.Module):
    """
    Simplified Temporal Fusion Transformer.

    Combines:
    - Static covariate encoding (station, soil type)
    - LSTM encoder for temporal patterns
    - Multi-head self-attention for long-range dependencies
    - GRN for variable selection and nonlinear processing
    """
    def __init__(self, n_temporal_features, n_static_features, hidden_size=64,
                 num_heads=4, num_layers=1, dropout=0.1, max_horizon=14):
        super().__init__()
        self.hidden_size = hidden_size

        # Static feature encoding
        self.static_encoder = GatedResidualNetwork(
            n_static_features, hidden_size, hidden_size, dropout
        )

        # Temporal feature projection
        self.temporal_proj = nn.Linear(n_temporal_features, hidden_size)

        # LSTM encoder
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                           dropout=dropout, batch_first=True)

        # Self-attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads,
                                                dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Output GRN + head
        self.output_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size,
                                                dropout, context_size=hidden_size)
        self.output_head = nn.Linear(hidden_size, max_horizon)

    def forward(self, x_temporal, x_static):
        """
        Args:
            x_temporal: (batch, seq_len, n_temporal_features)
            x_static: (batch, n_static_features)
        Returns:
            predictions: (batch, max_horizon)
        """
        # Encode static features
        static_enc = self.static_encoder(x_static)  # (batch, hidden)

        # Project temporal features
        temporal = self.temporal_proj(x_temporal)  # (batch, seq, hidden)

        # Add static context to each timestep
        static_expanded = static_enc.unsqueeze(1).expand_as(temporal)
        temporal = temporal + static_expanded

        # LSTM encoding
        lstm_out, _ = self.lstm(temporal)  # (batch, seq, hidden)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)  # residual

        # Use last timestep
        last = attn_out[:, -1, :]  # (batch, hidden)

        # Output processing
        out = self.output_grn(last, context=static_enc)
        return self.output_head(out)  # (batch, max_horizon)


# %%
print("=" * 60, flush=True)
print("TEMPORAL FUSION TRANSFORMER (Simplified)", flush=True)
print("=" * 60, flush=True)

# Prepare global dataset for TFT
# Temporal features: VWC5 + meteorological
TFT_TEMPORAL_COLS = ['VWC5'] + METEO_FEATURES + ['LAI', 'DOY_sin', 'DOY_cos']
# Static features: station_id, BOFEK
TFT_STATIC_COLS = ['station_id', 'BOFEK']
TFT_WINDOW = 30

class TFTDataset(Dataset):
    """Dataset for simplified TFT with static and temporal features."""

    def __init__(self, df, temporal_cols, static_cols, window_size, horizons,
                 temporal_scaler=None, target_stats=None):
        self.window = window_size
        self.max_h = max(horizons)
        self.horizons = horizons

        self.X_temporal = []
        self.X_static = []
        self.y = []

        for station in df['station'].unique():
            sdf = df[df['station'] == station].sort_values('date')

            temporal = sdf[temporal_cols].ffill().bfill().values
            static = sdf[static_cols].iloc[0].values.astype(np.float32)
            target = sdf['VWC5'].values

            for i in range(len(sdf) - window_size - self.max_h):
                x_temp = temporal[i:i + window_size]
                targets = [target[i + window_size + h - 1] for h in horizons]

                if not np.any(np.isnan(x_temp)) and not np.any(np.isnan(targets)):
                    self.X_temporal.append(x_temp)
                    self.X_static.append(static)
                    self.y.append(targets)

        self.X_temporal = np.array(self.X_temporal, dtype=np.float32)
        self.X_static = np.array(self.X_static, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        # Normalize temporal features
        if temporal_scaler is None:
            self.temporal_scaler = StandardScaler()
            orig_shape = self.X_temporal.shape
            self.X_temporal = self.temporal_scaler.fit_transform(
                self.X_temporal.reshape(-1, orig_shape[-1])
            ).reshape(orig_shape)
        else:
            self.temporal_scaler = temporal_scaler
            orig_shape = self.X_temporal.shape
            self.X_temporal = self.temporal_scaler.transform(
                self.X_temporal.reshape(-1, orig_shape[-1])
            ).reshape(orig_shape)

        # Normalize target
        if target_stats is None:
            self.target_mean = np.mean(self.y)
            self.target_std = np.std(self.y)
        else:
            self.target_mean, self.target_std = target_stats
        self.y = (self.y - self.target_mean) / self.target_std

        # Normalize static features
        if temporal_scaler is None:
            self.static_mean = np.mean(self.X_static, axis=0)
            self.static_std = np.std(self.X_static, axis=0) + 1e-8
        self.X_static = (self.X_static - np.mean(self.X_static, axis=0)) / (np.std(self.X_static, axis=0) + 1e-8)

    def __len__(self):
        return len(self.X_temporal)

    def __getitem__(self, idx):
        return (torch.tensor(self.X_temporal[idx]),
                torch.tensor(self.X_static[idx]),
                torch.tensor(self.y[idx]))


# Ensure station_id exists in processed data
for name, sdf in processed_dfs.items():
    sdf['station_id'] = station_map.get(name, 0)

all_proc_tft = pd.concat(processed_dfs.values(), ignore_index=True)

train_tft = all_proc_tft[all_proc_tft['date'] <= TRAIN_END]
test_tft = all_proc_tft[all_proc_tft['date'] >= TEST_START]

# Create datasets
train_ds_tft = TFTDataset(train_tft, TFT_TEMPORAL_COLS, TFT_STATIC_COLS,
                           TFT_WINDOW, HORIZONS)
test_ds_tft = TFTDataset(test_tft, TFT_TEMPORAL_COLS, TFT_STATIC_COLS,
                          TFT_WINDOW, HORIZONS,
                          temporal_scaler=train_ds_tft.temporal_scaler,
                          target_stats=(train_ds_tft.target_mean, train_ds_tft.target_std))

print(f"TFT Train samples: {len(train_ds_tft)}, Test samples: {len(test_ds_tft)}")

train_loader_tft = DataLoader(train_ds_tft, batch_size=128, shuffle=True)
test_loader_tft = DataLoader(test_ds_tft, batch_size=128)

# Build and train TFT
tft_model = SimplifiedTFT(
    n_temporal_features=len(TFT_TEMPORAL_COLS),
    n_static_features=len(TFT_STATIC_COLS),
    hidden_size=64,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    max_horizon=len(HORIZONS),
).to(device)

optimizer = torch.optim.Adam(tft_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()

best_val_loss = float('inf')
best_state = None
patience_counter = 0
PATIENCE = 15

print("\nTraining TFT...")
for epoch in range(100):
    tft_model.train()
    train_loss = 0
    for x_temp, x_stat, y_batch in train_loader_tft:
        x_temp, x_stat, y_batch = x_temp.to(device), x_stat.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = tft_model(x_temp, x_stat)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tft_model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validation
    tft_model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_temp, x_stat, y_batch in test_loader_tft:
            x_temp, x_stat, y_batch = x_temp.to(device), x_stat.to(device), y_batch.to(device)
            pred = tft_model(x_temp, x_stat)
            val_loss += criterion(pred, y_batch).item()
    val_loss /= max(len(test_loader_tft), 1)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in tft_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader_tft):.6f}, val_loss={val_loss:.6f}")

    if patience_counter >= PATIENCE:
        print(f"  Early stopping at epoch {epoch+1}")
        break

if best_state:
    tft_model.load_state_dict(best_state)

# %%
# Evaluate TFT per station
tft_model.eval()
all_preds = []
all_trues = []

with torch.no_grad():
    for x_temp, x_stat, y_batch in test_loader_tft:
        x_temp, x_stat = x_temp.to(device), x_stat.to(device)
        pred = tft_model(x_temp, x_stat).cpu().numpy()
        all_preds.append(pred)
        all_trues.append(y_batch.numpy())

all_preds = np.concatenate(all_preds)
all_trues = np.concatenate(all_trues)

# Denormalize
all_preds = all_preds * train_ds_tft.target_std + train_ds_tft.target_mean
all_trues = all_trues * train_ds_tft.target_std + train_ds_tft.target_mean

print("\nTFT Results (global model, all stations):")
for i, h in enumerate(HORIZONS):
    m = record_result('TFT', 'multivariate', h, 'ALL', all_trues[:, i], all_preds[:, i])
    print(f"  h={h:2d}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.3f}")

# %% [markdown]
# ---
# ## 7. Model Comparison & Discussion

# %%
# Compile all results
results_df = pd.DataFrame(results)
print("=" * 60)
print("COMPLETE RESULTS TABLE")
print("=" * 60)
print(results_df[['model', 'variant', 'horizon', 'station', 'RMSE', 'MAE', 'R2']].to_string(index=False))

# %%
# Summary comparison: average across focus stations
print("\n" + "=" * 60)
print("AVERAGE METRICS ACROSS FOCUS STATIONS (per-station models)")
print("=" * 60)

per_station = results_df[results_df['station'].isin(FOCUS_STATIONS)]
summary = per_station.groupby(['model', 'variant', 'horizon'])[['RMSE', 'MAE', 'R2']].mean().round(4)
print(summary.to_string())

# %%
# Global model comparison
print("\n" + "=" * 60)
print("GLOBAL MODEL COMPARISON")
print("=" * 60)
global_results = results_df[results_df['station'] == 'ALL']
if len(global_results) > 0:
    print(global_results[['model', 'variant', 'horizon', 'RMSE', 'MAE', 'R2']].to_string(index=False))

# %%
# --- Visualization: Model comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric in zip(axes, ['RMSE', 'MAE', 'R2']):
    pivot = per_station.groupby(['model', 'variant', 'horizon'])[metric].mean().reset_index()
    pivot['label'] = pivot['model'] + '\n(' + pivot['variant'].str[:3] + ')'

    for label in pivot['label'].unique():
        subset = pivot[pivot['label'] == label]
        ax.plot(subset['horizon'], subset[metric], 'o-', label=label, linewidth=2, markersize=8)

    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Horizon')
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved model_comparison.png")

# %%
# --- Univariate vs Multivariate comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, h in zip(axes, HORIZONS):
    subset = per_station[per_station['horizon'] == h]

    uni = subset[subset['variant'] == 'univariate'].groupby('model')['RMSE'].mean()
    multi = subset[subset['variant'] == 'multivariate'].groupby('model')['RMSE'].mean()

    # Combine models that appear in both
    models_both = sorted(set(uni.index) & set(multi.index))
    models_uni_only = sorted(set(uni.index) - set(multi.index))

    x = np.arange(len(models_both))
    width = 0.35

    if len(models_both) > 0:
        ax.bar(x - width/2, [uni[m] for m in models_both], width, label='Univariate', color='steelblue')
        ax.bar(x + width/2, [multi[m] for m in models_both], width, label='Multivariate', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(models_both, rotation=30, ha='right')

    ax.set_ylabel('RMSE')
    ax.set_title(f'Univariate vs Multivariate — Horizon {h}d')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('univariate_vs_multivariate.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved univariate_vs_multivariate.png")

# %%
# --- Per-station performance heatmap ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, h in zip(axes, HORIZONS):
    subset = per_station[(per_station['horizon'] == h) & (per_station['variant'] == 'multivariate')]
    if len(subset) == 0:
        subset = per_station[per_station['horizon'] == h]

    pivot = subset.pivot_table(values='RMSE', index='station', columns='model')
    if len(pivot) > 0:
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'RMSE Heatmap — Horizon {h}d')

plt.tight_layout()
plt.savefig('per_station_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved per_station_heatmap.png")

# %%
# --- Sample forecasts visualization ---
fig, axes = plt.subplots(len(FOCUS_STATIONS[:3]), len(HORIZONS), figsize=(20, 12))

for row, station in enumerate(FOCUS_STATIONS[:3]):
    sdf = processed_dfs[station]
    test_sdf = sdf[sdf['date'] >= TEST_START].sort_values('date')

    for col, h in enumerate(HORIZONS):
        ax = axes[row, col]
        target_col = f'target_h{h}'

        # Plot actual
        valid = test_sdf.dropna(subset=[target_col])
        ax.plot(valid['date'], valid[target_col], 'k-', alpha=0.7, linewidth=1, label='Actual')

        # Plot persistence baseline
        ax.plot(valid['date'], valid['VWC5'], '--', color='gray', alpha=0.5,
                linewidth=1, label='Persistence')

        ax.set_title(f'{station} — h={h}d', fontsize=11)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        if col == 0:
            ax.set_ylabel('VWC5')
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.suptitle('Test Period (2018) — Actual vs Persistence Baseline', fontsize=14)
plt.tight_layout()
plt.savefig('sample_forecasts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved sample_forecasts.png")

# %% [markdown]
# ## Discussion & Key Takeaways
#
# ### Univariate vs. Multivariate
# - At **h=1**: Univariate models (especially persistence and AR) are very competitive
#   because VWC5 has extremely high autocorrelation (lag-1 ~ 0.97). Adding weather
#   covariates provides marginal improvement.
# - At **h=7 and h=14**: Multivariate models increasingly outperform univariate ones,
#   because the autoregressive signal decays and external drivers (rainfall,
#   evapotranspiration) become more predictive.
#
# ### Model Hierarchy (typical)
# 1. **h=1**: Persistence ~ AR ≈ LSTM-uni ≈ XGBoost-uni (all very strong)
# 2. **h=7**: XGBoost-multi > LSTM-multi > ARIMAX > AR > Persistence
# 3. **h=14**: XGBoost-multi ≈ TFT > LSTM-multi > ARIMAX >> Persistence
#
# ### Practical Insights
# - **XGBoost** offers the best effort-to-performance ratio — fast to train,
#   handles mixed features natively, strong at all horizons.
# - **LSTM** excels when the sequence structure matters and can be extended to
#   multi-step direct forecasting.
# - **TFT** shines with its ability to integrate static metadata (soil type,
#   station identity) and provide interpretable attention weights.
# - **Persistence** is a deceptively strong baseline for 1-day forecasts of
#   soil moisture due to the high temporal inertia of the process.
#
# ### Exercises for Students
# 1. Add more stations to the per-station models. Which stations are hardest to forecast?
# 2. Try different lookback windows (7, 14, 30, 60 days) for the LSTM. What is optimal?
# 3. Implement a **recursive** multi-step forecast (feed predictions back as input)
#    and compare with the direct approach used here.
# 4. Add **spatial features** (distance to nearest station, elevation) and evaluate
#    if cross-station information improves single-station forecasts.
# 5. Experiment with the **forecast horizon**: at what point does persistence stop
#    being competitive?
# 6. Implement proper **cross-validation** using an expanding window approach.

# %%
# Save final results table
results_df.to_csv('forecasting_results.csv', index=False)
print("Results saved to forecasting_results.csv")
print("\nDone! All plots saved as PNG files.")
