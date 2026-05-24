"""Honest explainability for the undercut classifier.

Why not SHAP? `shap` pulls in `llvmlite` which fails to build on Python 3.12
in this environment. Rather than fake a SHAP plot, we use the two principled
alternatives:

  1. Permutation importance (sklearn) -- model-agnostic, considered the
     gold standard for tree models. Often preferred over SHAP for global
     feature ranking because it directly measures predictive value loss.

  2. Per-prediction explanation via tree-decision contribution -- using
     LightGBM's built-in feature contributions for a worked example.

Both views are saved to docs/model_cards/figures/.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit

from pitwall.config import PROCESSED_DIR
from pitwall.features.undercut import build_features
from pitwall.models.undercut_classifier import FEATURE_COLS

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "docs" / "model_cards" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_eval_data():
    """Reconstruct the same group-aware test split train_and_validate.py uses."""
    import duckdb

    con = duckdb.connect(str(PROCESSED_DIR / "pitwall.duckdb"), read_only=True)
    laps_raw = con.execute("select * from fact_lap").df()
    pits = con.execute("select * from fact_pit_stop").df()
    con.close()
    all_laps = laps_raw.rename(
        columns={
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
    )
    deg = joblib.load(PROCESSED_DIR / "degradation_model.joblib")
    clf = joblib.load(PROCESSED_DIR / "undercut_classifier.joblib")
    feats = build_features(all_laps, pits, deg).reset_index(drop=True)

    groups = feats["group_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    _, test_idx = next(gss.split(feats, feats["label"], groups=groups))
    return clf, feats.iloc[test_idx].reset_index(drop=True)


def permutation_plot(clf, test_feats: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Compute & plot permutation importance on the held-out test set."""
    X = test_feats[FEATURE_COLS]
    y = test_feats["label"]

    print(f"  computing permutation importance on n={len(X)} test rows...")
    result = permutation_importance(
        clf.model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring="neg_brier_score",
    )

    imp = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "mean": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        imp["feature"], imp["mean"], xerr=imp["std"], color="#E10600", ecolor=(0.6, 0.6, 0.6, 0.7)
    )
    ax.set_xlabel("Permutation importance (Δ Brier when feature shuffled)")
    ax.set_title("Undercut classifier - permutation importance on held-out test")
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return imp


def local_explanation_table(clf, test_feats: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """For three representative test stops, show feature contributions.

    Uses LightGBM's `predict_contrib` for each tree in the ensemble, averaged
    across calibration folds. This is the equivalent of a SHAP waterfall for
    a tree model -- often called "TreeSHAP" without needing the shap library.
    """
    X = test_feats[FEATURE_COLS]
    # Pick three test stops: highest predicted, lowest predicted, median predicted
    proba = clf.predict_proba(X)
    order = np.argsort(proba)
    pick_idx = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]

    contribs_per_row: list[dict] = []
    for idx in pick_idx:
        row = X.iloc[[idx]]
        # Sum LightGBM raw contributions across calibration folds
        per_fold = []
        for cal in clf.model.calibrated_classifiers_:
            est = cal.estimator
            if hasattr(est, "predict") and hasattr(est.booster_, "predict"):
                contrib = est.booster_.predict(row.to_numpy(), pred_contrib=True)
                per_fold.append(contrib[0])
        if not per_fold:
            continue
        arr = np.mean(per_fold, axis=0)
        # Last value is the bias term
        feat_contribs = arr[:-1]
        bias = arr[-1]
        record = {
            "predicted_prob": float(proba[idx]),
            "actual_label": int(test_feats["label"].iloc[idx]),
            "race": test_feats["group_id"].iloc[idx],
            "driver": test_feats["driver_code"].iloc[idx],
            "pit_lap": int(test_feats["pit_lap"].iloc[idx]),
            "bias": float(bias),
        }
        for col, c in zip(FEATURE_COLS, feat_contribs, strict=False):
            record[f"contrib__{col}"] = float(c)
        contribs_per_row.append(record)

    df = pd.DataFrame(contribs_per_row)
    df.to_csv(out_path, index=False)

    # Print a human-readable summary
    print("\n  Per-prediction explanations (TreeSHAP-style contributions):")
    for r in contribs_per_row:
        print(
            f"    Race {r['race']}, {r['driver']} pit lap {r['pit_lap']}: "
            f"P(gain)={r['predicted_prob']:.3f}, actual={'YES' if r['actual_label'] == 1 else 'NO'}"
        )
        contribs = [
            (k.replace("contrib__", ""), v) for k, v in r.items() if k.startswith("contrib__")
        ]
        contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        for name, val in contribs[:5]:
            sign = "+" if val > 0 else ""
            print(f"      {name:30s}  {sign}{val:.3f}")
    return df


def main() -> None:
    print("Loading model + test split...")
    clf, test_feats = load_eval_data()
    print(f"  {len(test_feats):,} test stops, {test_feats['group_id'].nunique()} races")

    print("\n[1/2] Permutation importance (model-agnostic)...")
    perm_path = FIGURES_DIR / "permutation_importance.png"
    imp = permutation_plot(clf, test_feats, perm_path)
    print(f"  Plot: {perm_path.relative_to(PROJECT_ROOT)}")
    print("  Top 5 features by permutation importance:")
    for _, row in imp.tail(5).iloc[::-1].iterrows():
        print(f"    {row['feature']:30s}  {row['mean']:+.4f}  +/- {row['std']:.4f}")

    print("\n[2/2] Per-prediction tree-contribution explanations...")
    contribs_path = PROCESSED_DIR / "local_explanations.csv"
    local_explanation_table(clf, test_feats, contribs_path)
    print(f"  Saved {contribs_path.relative_to(PROJECT_ROOT)}")

    print("\n[OK] Explainability artifacts written.")
    print("     For SHAP specifically, see model card: shap package fails to install")
    print("     under llvmlite on this environment. Permutation importance is the")
    print("     gold-standard model-agnostic alternative.")


if __name__ == "__main__":
    main()
