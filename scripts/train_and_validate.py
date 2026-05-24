"""End-to-end ML pipeline validation on real ingested data.

This is the canonical "does the whole pipeline still work" check. It produces
every metric quoted in the README and the model cards.

What it does:
  1. Loads fact_lap / fact_stint / fact_pit_stop from DuckDB
  2. Fits the DegradationModel with within-circuit + leave-one-circuit-out CV
  3. Builds REAL undercut features (no hardcoded constants)
  4. Splits with GroupKFold on (year, round_num) -- no race-day leakage
  5. Trains FOUR models: constant baseline, threshold rule, logistic regression,
     calibrated LightGBM
  6. Saves calibration plot + feature-importance plot to docs/model_cards/figures/
  7. Logs every run to MLflow (local SQLite backend)
  8. Writes model_comparison.json with the full table

Run with: uv run python scripts/train_and_validate.py
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from pitwall.config import PROCESSED_DIR
from pitwall.features.undercut import build_features
from pitwall.models.degradation_curve import DegradationModel
from pitwall.models.pit_cost import circuit_pit_cost
from pitwall.models.undercut_classifier import FEATURE_COLS, UndercutClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = PROCESSED_DIR / "pitwall.duckdb"
FIGURES_DIR = PROJECT_ROOT / "docs" / "model_cards" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR = PROCESSED_DIR / "mlruns"
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file:{MLFLOW_DIR.as_posix()}")
mlflow.set_experiment("pitwall-undercut")


# ============================================================
# Data loading
# ============================================================


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    laps = con.execute("select * from fact_lap").df()
    stints = con.execute("select * from fact_stint").df()
    pits = con.execute("select * from fact_pit_stop").df()
    con.close()
    return laps, stints, pits


_RENAME = {
    "year": "Year",
    "round_num": "Round",
    "driver_code": "Driver",
    "circuit_name": "CircuitName",
    "stint": "Stint",
    "lap_number": "LapNumber",
    "stint_position": "StintPosition",
    "compound": "Compound",
    "lap_time_s": "LapTimeSeconds",
    "lap_time_fuel_corrected_s": "LapTimeFuelCorrected",
    "is_clean_lap": "IsCleanLap",
    "position": "Position",
}


def harmonise_all_laps(laps: pd.DataFrame) -> pd.DataFrame:
    return laps.rename(columns=_RENAME)


def harmonise_clean_laps(all_laps_renamed: pd.DataFrame) -> pd.DataFrame:
    return all_laps_renamed[
        (all_laps_renamed["IsCleanLap"]) & (all_laps_renamed["StintPosition"] > 1)
    ]


def harmonise_stints(stints: pd.DataFrame) -> pd.DataFrame:
    return stints.rename(
        columns={
            "year": "Year",
            "round_num": "Round",
            "driver_code": "Driver",
            "stint": "Stint",
            "deg_fit_r2": "DegFitR2",
        }
    )


# ============================================================
# Degradation -- with leave-one-circuit-out CV
# ============================================================


def fit_degradation_loco(laps: pd.DataFrame, stints: pd.DataFrame) -> dict:
    """Leave-one-circuit-out CV across all 33 circuits + within-circuit holdout."""

    # Within-circuit holdout: random 20% of stints
    rng = np.random.default_rng(42)
    stint_keys = laps[["Round", "Driver", "Stint"]].drop_duplicates().reset_index(drop=True)
    holdout_mask = rng.random(len(stint_keys)) < 0.2
    holdout_keys = stint_keys[holdout_mask]
    is_holdout = (
        laps.merge(holdout_keys, on=["Round", "Driver", "Stint"], how="left", indicator=True)
        ._merge.eq("both")
        .to_numpy()
    )
    train = laps[~is_holdout]
    test = laps[is_holdout]
    within_model = DegradationModel(min_samples_per_curve=20).fit(stints, train)
    within_mae = within_model.mae(stints, test)

    # LOCO across all circuits with at least 200 test laps
    loco_results = []
    circuits = laps["CircuitName"].dropna().unique()
    for circuit in circuits:
        train_c = laps[laps["CircuitName"] != circuit]
        test_c = laps[laps["CircuitName"] == circuit]
        if len(test_c) < 200:
            continue
        m = DegradationModel(min_samples_per_curve=20).fit(stints, train_c)
        loco_results.append(
            {
                "circuit": circuit,
                "n_test_laps": len(test_c),
                "mae_s": float(m.mae(stints, test_c)),
            }
        )
    loco = pd.DataFrame(loco_results).sort_values("mae_s")

    # Production model trained on everything
    prod = DegradationModel(min_samples_per_curve=20).fit(stints, laps)

    return {
        "within_mae_s": within_mae,
        "n_test_within": int(is_holdout.sum()),
        "loco_table": loco,
        "loco_median_mae_s": float(loco["mae_s"].median()) if not loco.empty else float("nan"),
        "loco_iqr": (
            (float(loco["mae_s"].quantile(0.25)), float(loco["mae_s"].quantile(0.75)))
            if not loco.empty
            else (float("nan"), float("nan"))
        ),
        "prod_model": prod,
        "prod_n_curves": len(prod.curves),
        "prod_n_global": len(prod.global_curves),
    }


# ============================================================
# Baselines for the undercut classifier
# ============================================================


@dataclass
class ModelResult:
    name: str
    auc: float
    brier: float
    log_loss_value: float
    proba_test: np.ndarray
    feature_cols: list[str]


def evaluate_baselines(
    feats: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> list[ModelResult]:
    """Three baselines for honest comparison against the LightGBM."""
    X = feats[FEATURE_COLS]
    y = feats["label"].to_numpy()

    results: list[ModelResult] = []

    # --- Baseline 1: constant predictor (always predict base rate) ---
    base_rate = float(y[train_idx].mean())
    proba_const = np.full(len(test_idx), base_rate)
    results.append(
        ModelResult(
            name="constant (base rate)",
            auc=0.5,
            brier=float(brier_score_loss(y[test_idx], proba_const)),
            log_loss_value=float(log_loss(y[test_idx], proba_const, labels=[0, 1])),
            proba_test=proba_const,
            feature_cols=[],
        )
    )

    # --- Baseline 2: single-feature threshold rule on tyre_age ---
    # Pick the threshold that maximises train AUC, then evaluate on test
    best_auc, best_thresh = 0.5, 15
    for t in range(5, 35):
        # rule: predict positive if tyre_age >= t and laps_remaining > 5
        pred = (
            (
                (feats["tyre_age"].iloc[train_idx] >= t)
                & (feats["laps_remaining"].iloc[train_idx] > 5)
            )
            .astype(float)
            .to_numpy()
        )
        if len(np.unique(pred)) < 2:
            continue
        auc = roc_auc_score(y[train_idx], pred)
        if auc > best_auc:
            best_auc, best_thresh = auc, t
    pred_test = (
        (
            (feats["tyre_age"].iloc[test_idx] >= best_thresh)
            & (feats["laps_remaining"].iloc[test_idx] > 5)
        )
        .astype(float)
        .to_numpy()
    )
    auc_thr = float(roc_auc_score(y[test_idx], pred_test)) if len(np.unique(pred_test)) > 1 else 0.5
    # Smooth to a probability for Brier
    proba_thr = np.where(pred_test == 1, 0.5 + (base_rate / 2), base_rate / 2)
    results.append(
        ModelResult(
            name=f"threshold rule (tyre_age >= {best_thresh})",
            auc=auc_thr,
            brier=float(brier_score_loss(y[test_idx], proba_thr)),
            log_loss_value=float(log_loss(y[test_idx], proba_thr, labels=[0, 1])),
            proba_test=proba_thr,
            feature_cols=["tyre_age", "laps_remaining"],
        )
    )

    # --- Baseline 3: logistic regression with standard scaling ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_test = scaler.transform(X.iloc[test_idx])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y[train_idx])
    proba_lr = lr.predict_proba(X_test)[:, 1]
    results.append(
        ModelResult(
            name="logistic regression (balanced)",
            auc=float(roc_auc_score(y[test_idx], proba_lr)),
            brier=float(brier_score_loss(y[test_idx], proba_lr)),
            log_loss_value=float(log_loss(y[test_idx], proba_lr, labels=[0, 1])),
            proba_test=proba_lr,
            feature_cols=FEATURE_COLS,
        )
    )

    return results


# ============================================================
# Calibration + feature importance plots
# ============================================================


def save_calibration_plot(
    y_true: np.ndarray, proba_dict: dict[str, np.ndarray], out_path: Path
) -> dict[str, float]:
    """Reliability diagram for several models on the same axes.

    Returns ECE per model.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ece_dict: dict[str, float] = {}

    for name, proba in proba_dict.items():
        # Use 10 bins; clip degenerate cases
        try:
            frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        except ValueError:
            continue
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        ece_dict[name] = ece
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{name}  (ECE={ece:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive frequency")
    ax.set_title("Reliability diagram (10 quantile bins)")
    ax.set_xlim([0, max(0.3, max(p.max() for p in proba_dict.values()))])
    ax.set_ylim([0, max(0.3, max(p.max() for p in proba_dict.values()))])
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return ece_dict


def save_feature_importance_plot(clf: UndercutClassifier, out_path: Path) -> dict[str, float]:
    """Permutation-based feature importance via the underlying LightGBM gain.

    SHAP is also wired in (optional extra) but the gain-based view is sufficient
    for the model card and avoids the SHAP dependency for the CI path.
    """
    if clf.model is None:
        return {}
    # CalibratedClassifierCV stacks estimators; average their feature_importances_
    importances = []
    for cal in clf.model.calibrated_classifiers_:
        est = cal.estimator
        if hasattr(est, "feature_importances_"):
            importances.append(est.feature_importances_)
    if not importances:
        return {}
    importance_mean = np.mean(importances, axis=0)
    order = np.argsort(importance_mean)[::-1]
    sorted_features = [clf.feature_cols[i] for i in order]
    sorted_values = importance_mean[order]
    result = {f: float(v) for f, v in zip(sorted_features, sorted_values, strict=False)}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sorted_features[::-1], sorted_values[::-1], color="#E10600")
    ax.set_xlabel("LightGBM gain importance (averaged across calibration folds)")
    ax.set_title("Undercut classifier - feature importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return result


# ============================================================
# Main pipeline
# ============================================================


def main() -> None:
    print("=" * 70)
    print("Pit Wall Intelligence -- End-to-End ML Pipeline Validation")
    print("=" * 70)

    # ---- Load ----
    print("\n[1/6] Loading data from DuckDB...")
    laps_raw, stints_raw, pits_raw = load_data()
    all_laps = harmonise_all_laps(laps_raw)
    laps = harmonise_clean_laps(all_laps)
    stints = harmonise_stints(stints_raw)
    print(f"  fact_lap (all):    {len(all_laps):,} rows")
    print(
        f"  fact_lap (clean):  {len(laps):,} rows across {laps['CircuitName'].nunique()} circuits"
    )
    print(f"  fact_stint:        {len(stints):,} rows")
    print(f"  fact_pit_stop:     {len(pits_raw):,} rows")

    # ---- Degradation with LOCO ----
    print("\n[2/6] Tyre degradation model with LOCO cross-validation...")
    with mlflow.start_run(run_name="degradation"):
        deg = fit_degradation_loco(laps, stints)
        print(
            f"  Within-circuit MAE: {deg['within_mae_s']:.3f} s  ({deg['n_test_within']} test stint-keys)"
        )
        if deg["loco_median_mae_s"] is not None and not np.isnan(deg["loco_median_mae_s"]):
            low, high = deg["loco_iqr"]
            print(
                f"  LOCO MAE: median {deg['loco_median_mae_s']:.3f}s, IQR [{low:.3f}, {high:.3f}]"
            )
            print("  LOCO worst (top 3 hardest circuits):")
            for _, row in deg["loco_table"].tail(3).iterrows():
                print(
                    f"    {row['circuit']:30s}  MAE {row['mae_s']:5.2f}s  (n={row['n_test_laps']})"
                )
        print(f"  Production model: {deg['prod_n_curves']} per-circuit curves")
        # MLflow logging
        mlflow.log_metric("within_circuit_mae_s", deg["within_mae_s"])
        mlflow.log_metric("loco_median_mae_s", deg["loco_median_mae_s"])
        mlflow.log_metric("n_circuits", deg["prod_n_curves"])
        deg["loco_table"].to_csv(PROCESSED_DIR / "loco_degradation_mae.csv", index=False)
        mlflow.log_artifact(str(PROCESSED_DIR / "loco_degradation_mae.csv"))
        joblib.dump(deg["prod_model"], PROCESSED_DIR / "degradation_model.joblib")
        mlflow.log_artifact(str(PROCESSED_DIR / "degradation_model.joblib"))

    # ---- Pit cost ----
    print("\n[3/6] Pit-cost summary by circuit (with bootstrap CI)...")
    cost_df = pits_raw.rename(columns={"circuit_name": "CircuitName", "pit_loss_s": "PitLossS"})
    cost = circuit_pit_cost(cost_df, n_bootstrap=1000)
    print(f"  {len(cost)} circuits, range:")
    print(f"    fastest:  {cost.iloc[0]['CircuitName']:30s}  {cost.iloc[0]['MedianPitLossS']:.2f}s")
    print(
        f"    slowest:  {cost.iloc[-1]['CircuitName']:30s}  {cost.iloc[-1]['MedianPitLossS']:.2f}s"
    )
    cost.to_csv(PROCESSED_DIR / "circuit_pit_cost.csv", index=False)

    # ---- Real undercut features ----
    print("\n[4/6] Building REAL undercut features (no hardcoded constants)...")
    feats = build_features(all_laps, pits_raw, deg["prod_model"])
    feats = feats.reset_index(drop=True)
    print(f"  examples:        {len(feats):,} pit stops")
    print(f"  positive rate:   {feats['label'].mean():.3f}")
    print(f"  unique races:    {feats['group_id'].nunique()}")
    print(
        f"  gap_ahead range: {feats['gap_ahead_s'].min():.2f} .. {feats['gap_ahead_s'].quantile(0.95):.2f}"
    )
    print(
        f"  deg_slope range: {feats['current_deg_slope'].quantile(0.05):.3f} .. {feats['current_deg_slope'].quantile(0.95):.3f}"
    )

    # ---- Group split + baselines + LightGBM ----
    print("\n[5/6] Group-aware train/test split + baselines + LightGBM...")
    groups = feats["group_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(feats, feats["label"], groups=groups))
    n_train_races = len(np.unique(groups[train_idx]))
    n_test_races = len(np.unique(groups[test_idx]))
    print(f"  train races: {n_train_races}, test races: {n_test_races}, no overlap.")

    baselines = evaluate_baselines(feats, train_idx, test_idx)

    with mlflow.start_run(run_name="undercut-lightgbm"):
        clf = UndercutClassifier()
        clf.fit(feats[FEATURE_COLS].iloc[train_idx], feats["label"].iloc[train_idx])
        clf_proba = clf.predict_proba(feats[FEATURE_COLS].iloc[test_idx])
        clf_y = feats["label"].iloc[test_idx].to_numpy()
        clf_result = ModelResult(
            name="LightGBM (isotonic calibrated)",
            auc=float(roc_auc_score(clf_y, clf_proba)),
            brier=float(brier_score_loss(clf_y, clf_proba)),
            log_loss_value=float(log_loss(clf_y, clf_proba, labels=[0, 1])),
            proba_test=clf_proba,
            feature_cols=FEATURE_COLS,
        )

        # 5-fold GroupKFold CV for stability
        gkf = GroupKFold(n_splits=5)
        cv_aucs = []
        for tr, te in gkf.split(feats, feats["label"], groups=groups):
            cv_clf = UndercutClassifier()
            cv_clf.fit(feats[FEATURE_COLS].iloc[tr], feats["label"].iloc[tr])
            cv_proba = cv_clf.predict_proba(feats[FEATURE_COLS].iloc[te])
            cv_y = feats["label"].iloc[te].to_numpy()
            if len(np.unique(cv_y)) > 1:
                cv_aucs.append(roc_auc_score(cv_y, cv_proba))
        cv_auc_mean = float(np.mean(cv_aucs)) if cv_aucs else float("nan")
        cv_auc_std = float(np.std(cv_aucs)) if cv_aucs else float("nan")
        print(f"  5-fold GroupKFold AUC: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}")

        # MLflow logging for the LightGBM run
        mlflow.log_params(
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "calibration": "isotonic_5fold_cv",
                "split": "GroupShuffleSplit on (year, round_num)",
                "n_features": len(FEATURE_COLS),
            }
        )
        mlflow.log_metric("test_auc", clf_result.auc)
        mlflow.log_metric("test_brier", clf_result.brier)
        mlflow.log_metric("test_log_loss", clf_result.log_loss_value)
        mlflow.log_metric("cv_auc_mean", cv_auc_mean)
        mlflow.log_metric("cv_auc_std", cv_auc_std)
        mlflow.log_metric("n_train", len(train_idx))
        mlflow.log_metric("n_test", len(test_idx))
        mlflow.log_metric("base_rate", float(feats["label"].mean()))

        joblib.dump(clf, PROCESSED_DIR / "undercut_classifier.joblib")
        mlflow.log_artifact(str(PROCESSED_DIR / "undercut_classifier.joblib"))

    # ---- Comparison table ----
    print("\n[6/6] Model comparison + plots...")
    all_results = [*baselines, clf_result]
    print(f"  {'Model':40s}  {'AUC':>7s}  {'Brier':>7s}  {'LogLoss':>8s}")
    print("  " + "-" * 70)
    for r in all_results:
        print(f"  {r.name:40s}  {r.auc:7.3f}  {r.brier:7.4f}  {r.log_loss_value:8.4f}")

    comparison = {
        "base_rate": float(feats["label"].mean()),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_races": int(n_train_races),
        "n_test_races": int(n_test_races),
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "models": [
            {
                "name": r.name,
                "auc": r.auc,
                "brier": r.brier,
                "log_loss": r.log_loss_value,
            }
            for r in all_results
        ],
    }
    (PROCESSED_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2))

    # Calibration plot for the LightGBM and logistic regression
    cal_path = FIGURES_DIR / "calibration.png"
    ece = save_calibration_plot(
        feats["label"].iloc[test_idx].to_numpy(),
        {
            "LightGBM (isotonic)": clf_result.proba_test,
            "Logistic regression": baselines[2].proba_test,
        },
        cal_path,
    )
    print(f"  Calibration plot: {cal_path.relative_to(PROJECT_ROOT)}  ECE={ece}")

    # Feature importance
    fi_path = FIGURES_DIR / "feature_importance.png"
    fi = save_feature_importance_plot(clf, fi_path)
    if fi:
        print(f"  Feature importance: {fi_path.relative_to(PROJECT_ROOT)}")
        print(f"    Top 5: {list(fi.items())[:5]}")

    print("\n" + "=" * 70)
    print("[OK] Pipeline complete. See data/processed/model_comparison.json for the full table.")
    print(f"     MLflow runs:  uv run mlflow ui --backend-store-uri file:{MLFLOW_DIR.as_posix()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
