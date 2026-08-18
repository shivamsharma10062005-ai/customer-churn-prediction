"""Train churn-prediction models as a production-grade, versioned bundle.

Key differences from the original script:
  * Stratified train/val/test split (val is used ONLY for calibration and
    threshold selection, never for model fitting, never for final metrics).
  * SMOTE is applied to the train split only; the class-prior distortion it
    causes is corrected with isotonic calibration fit on the *unresampled*
    validation split.
  * The decision threshold is chosen to maximize expected profit (intervention
    cost vs. retained value), not assumed to be 0.5.
  * Every artifact is versioned and described by a model_card.json (checksum,
    metrics, reference profile, baseline distribution) so predictions are
    traceable and drift can be monitored.

Usage:
    python train_churn.py [--version 2.0.0] [--value 120] [--cost 10]
                          [--effect 0.35] [--dataset-version synth-1.0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (auc, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from app import features as feat

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "telco_churn.csv"
ARTIFACTS = HERE / "artifacts"
MODEL_DIR = ARTIFACTS / "model"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["num_addons"] = feat.derive_num_addons_series(df)
    df = df.drop(columns=["customerID"]).dropna().reset_index(drop=True)
    return df


def build_preprocessor():
    cat = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num = Pipeline([("scaler", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("num", num, feat.NUMERIC),
            ("cat", cat, feat.CATEGORICAL),
        ]
    )


def _models() -> dict[str, object]:
    return {
        "logistic": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42),
    }


def _calibrate(base, X_val_pre, y_val):
    raw_val = base.predict_proba(X_val_pre)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    return iso


def _ece(y_true, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = indices == b
        if mask.sum() == 0:
            continue
        ece += np.abs(prob[mask].mean() - y_true[mask].mean()) * (mask.sum() / len(y_true))
    return float(ece)


def _optimal_threshold(prob, y, value, cost, effect):
    best_t, best_profit = 0.5, -np.inf
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (prob >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        profit = tp * value * effect - fp * cost
        if profit > best_profit:
            best_profit, best_t = profit, float(t)
    return best_t, float(best_profit)


def _top_decile_recall(y_true, prob):
    n = len(y_true)
    k = max(1, int(np.ceil(n * 0.10)))
    top = np.argsort(prob)[::-1][:k]
    churners = int(y_true.sum())
    caught = int(y_true.iloc[top].sum()) if hasattr(y_true, "iloc") else int(y_true[top].sum())
    return (caught / churners) if churners else 0.0, float(y_true.iloc[top].mean())


def _reference_profile(X_tr: pd.DataFrame) -> dict:
    profile = {}
    for col in feat.CATEGORICAL:
        profile[col] = X_tr[col].mode().iloc[0]
    for col in feat.NUMERIC:
        profile[col] = float(X_tr[col].median())
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Train versioned churn model bundle")
    parser.add_argument("--version", default="2.0.0", help="model version, e.g. 2.0.0")
    parser.add_argument("--dataset-version", default="synth-1.0")
    parser.add_argument("--value", type=float, default=120.0, help="retained customer value ($)")
    parser.add_argument("--cost", type=float, default=10.0, help="intervention cost ($)")
    parser.add_argument("--effect", type=float, default=0.35, help="intervention effectiveness")
    args = parser.parse_args()

    df = load_data()
    X = df.drop(columns=[feat.TARGET])
    y = (df[feat.TARGET] == "Yes").astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)

    # stratified splits: train/val/test. val is the calibration & threshold set.
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=42, stratify=y_rest)

    pre = build_preprocessor()
    X_tr_pre = pre.fit_transform(X_tr)
    X_val_pre = pre.transform(X_val)
    X_te_pre = pre.transform(X_te)

    X_tr_bal, y_tr_bal = SMOTE(random_state=42).fit_resample(X_tr_pre, y_tr)

    outcomes = {}
    for name, clf in _models().items():
        clf.fit(X_tr_bal, y_tr_bal)
        iso = _calibrate(clf, X_val_pre, y_val)

        raw_val = clf.predict_proba(X_val_pre)[:, 1]
        raw_te = clf.predict_proba(X_te_pre)[:, 1]
        cal_val = iso.predict(raw_val)
        cal_te = iso.predict(raw_te)

        threshold, val_profit = _optimal_threshold(cal_val, y_val, args.value, args.cost, args.effect)
        pred_te = (cal_te >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred_te).ravel()

        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        prec_curve, rec_curve, _ = precision_recall_curve(y_te, cal_te)
        top10_recall, top10_rate = _top_decile_recall(y_te, cal_te)

        outcomes[name] = {
            "algorithm": name,
            "roc_auc": float(roc_auc_score(y_te, cal_te)),
            "pr_auc": float(auc(rec_curve, prec_curve)),
            "brier": float(brier_score_loss(y_te, cal_te)),
            "ece": _ece(np.asarray(y_te), np.asarray(cal_te)),
            "threshold": threshold,
            "val_profit": val_profit,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "top_decile_recall": top10_recall,
            "top_decile_churn_rate": top10_rate,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "model": clf,
            "isotonic": iso,
            "raw_te": raw_te,
            "cal_te": cal_te,
            "cal_val": cal_val,
            "y_test": y_te,
        }

    # Business-aware model selection: best ranking quality on validation
    # (PR-AUC). Thresholds are chosen later by expected profit.
    for name in outcomes:
        p_v, r_v, _ = precision_recall_curve(y_val, outcomes[name]["cal_val"])
        outcomes[name]["pr_auc_val"] = float(auc(r_v, p_v))

    best_name = max(outcomes, key=lambda n: outcomes[n]["pr_auc_val"])
    best = outcomes[best_name]

    bundle_dir = MODEL_DIR / f"v{args.version}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    full_pipeline = Pipeline([
        ("pre", pre),
        ("clf", best["model"]),
    ])
    pipeline_path = bundle_dir / "pipeline.joblib"
    joblib_dump = lambda obj, path: joblib.dump(obj, path)  # noqa: E731
    joblib_dump(full_pipeline, pipeline_path)
    joblib_dump(best["isotonic"], bundle_dir / "isotonic.joblib")

    _save_plots(bundle_dir, outcomes, best_name)
    _save_calibration_plot(bundle_dir, best, y_te)

    card = {
        "model_version": f"v{args.version}",
        "feature_version": feat.FEATURE_VERSION,
        "dataset_version": args.dataset_version,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": best_name,
        "status": "champion",
        "pipeline_sha256": _sha256(pipeline_path),
        "best_model_selection": "max PR-AUC on held-out validation (thresholds from expected profit)",
        "business_parameters": {
            "retained_value": args.value,
            "intervention_cost": args.cost,
            "intervention_effect": args.effect,
        },
        "risk_thresholds": {
            "low": round(max(0.05, best["threshold"] * 0.4), 3),
            "medium": round(max(best["threshold"] * 0.4 + 0.01, best["threshold"]), 3),
            "high": round(min(0.95, best["threshold"] + 0.20), 3),
        },
        "metrics": {
            "roc_auc": round(best["roc_auc"], 4),
            "pr_auc": round(best["pr_auc"], 4),
            "brier": round(best["brier"], 4),
            "ece": round(best["ece"], 4),
            "precision": round(best["precision"], 4),
            "recall": round(best["recall"], 4),
            "f1": round(best["f1"], 4),
            "top_decile_recall": round(best["top_decile_recall"], 4),
            "threshold": best["threshold"],
        },
        "reference_profile": _reference_profile(X_tr),
        "baseline_mean_probability": round(float(np.asarray(best["cal_te"]).mean()), 6),
        "baseline_prediction_distribution": [round(float(p), 6) for p in best["cal_te"]],
        "all_models": {n: {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in o.items()
                           if k not in ("model", "isotonic", "raw_te", "cal_te", "cal_val", "y_test")}
                       for n, o in outcomes.items()},
    }

    card_path = bundle_dir / "model_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    (MODEL_DIR / "latest.json").write_text(
        json.dumps({"version": f"v{args.version}"}), encoding="utf-8")

    # Backward-compatible summary
    results_path = ARTIFACTS / "results.json"
    results_path.write_text(
        json.dumps({"best_model": best_name,
                    "model_version": f"v{args.version}",
                    "models": card["all_models"]}, indent=2),
        encoding="utf-8")

    print(f"\n=== BEST MODEL: {best_name} (threshold {best['threshold']:.2f}) ===")
    for name, o in outcomes.items():
        print(f"{name:14s} ROC={o['roc_auc']:.3f}  PR={o['pr_auc']:.3f}  Brier={o['brier']:.3f}  "
              f"ECE={o['ece']:.3f}  R@10%={o['top_decile_recall']:.3f}  profit={o['val_profit']:.0f}")
    print(f"Bundle written to {bundle_dir}")
    print(f"Calibrated probabilities: brier={best['brier']:.4f} ece={best['ece']:.4f}")


def _save_plots(bundle_dir: Path, outcomes: dict, best_name: str):
    best = outcomes[best_name]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, o in outcomes.items():
        prob = np.asarray(o["cal_te"])
        y_te = np.asarray(o["y_test"])
        fpr, tpr, _ = roc_curve(y_te, prob)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {o['roc_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8)
    ax[0].set(title="ROC curves (calibrated)", xlabel="FPR", ylabel="TPR")
    ax[0].legend(fontsize=8)
    p, r, _ = precision_recall_curve(np.asarray(best["y_test"]), np.asarray(best["cal_te"]))
    ax[1].plot(r, p, label=f"{best_name} (PR AUC {best['pr_auc']:.3f})")
    ax[1].set(title="Precision-Recall (best)", xlabel="Recall", ylabel="Precision")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(bundle_dir / "roc_pr_curves.png", dpi=120)
    plt.close(fig)

    tn, fp, fn, tp = best["tn"], best["fp"], best["fn"], best["tp"]
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Pred No", "Pred Yes"],
           yticklabels=["True No", "True Yes"],
           title=f"Confusion matrix (t={best['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(bundle_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def _save_calibration_plot(bundle_dir: Path, best: dict, y_te):
    prob = np.asarray(best["cal_te"])
    y = np.asarray(y_te)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, 9)
    means = []
    for b in range(10):
        mask = idx == b
        means.append(y[mask].mean() if mask.sum() else np.nan)
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.9)
    ax.plot(edges[:-1] + 0.05, means, marker="o", ms=4, color="#20D9FF")
    ax.set(title="Calibration curve", xlabel="Predicted probability", ylabel="Actual rate")
    fig.tight_layout()
    fig.savefig(bundle_dir / "calibration_curve.png", dpi=120)
    plt.close(fig)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
