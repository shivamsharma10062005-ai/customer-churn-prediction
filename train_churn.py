"""Train churn-prediction models on the synthetic telecom dataset.

Pipeline: preprocessing (OneHotEncoder) -> SMOTE resampling -> classifier.
Models: LogisticRegression, RandomForest, XGBoost.
Metrics: ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
Saves best model (as a self-contained sklearn pipeline) + plots + results.json.
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "telco_churn.csv")
MODEL_DIR = os.path.join(HERE, "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "Churn"
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # TotalCharges can contain blank strings -> numeric with NA fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df = df.drop(columns=["customerID"])
    df = df.dropna().reset_index(drop=True)
    return df


def build_preprocessor():
    cat = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num = Pipeline([("scaler", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("num", num, NUMERIC),
            ("cat", cat, CATEGORICAL),
        ]
    )


def train_eval(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == "Yes").astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    pre = build_preprocessor()
    X_tr_pre = pre.fit_transform(X_tr)
    X_te_pre = pre.transform(X_te)

    # SMOTE applied AFTER preprocessing, on train only — never touches test data
    X_tr_bal, y_tr_bal = SMOTE(random_state=42).fit_resample(X_tr_pre, y_tr)

    models = {
        "logistic": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_tr_bal, y_tr_bal)
        prob = clf.predict_proba(X_te_pre)[:, 1]
        pred = (prob >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_te, prob)
        prec, rec, _ = precision_recall_curve(y_te, prob)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

        results[name] = {
            "roc_auc": float(roc_auc_score(y_te, prob)),
            "pr_auc": float(auc(rec, prec)),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "model": clf,
            "roc": (fpr, tpr),
            "pr": (rec, prec),
        }
    return results, pre, X_te, y_te


def save_plots(results, best_name):
    fpr, tpr = results[best_name]["roc"]
    rec, prec = results[best_name]["pr"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, r in results.items():
        ax[0].plot(*r["roc"], label=f"{name} (AUC {r['roc_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8)
    ax[0].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend(fontsize=8)
    ax[1].plot(rec, prec, label=f"{best_name} (PR AUC {results[best_name]['pr_auc']:.3f})")
    ax[1].set(title="Precision-Recall (best model)", xlabel="Recall", ylabel="Precision")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "roc_pr_curves.png"), dpi=120)
    plt.close(fig)

    cm = np.array([[results[best_name]["tn"], results[best_name]["fp"]],
                   [results[best_name]["fn"], results[best_name]["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Pred No", "Pred Yes"],
           yticklabels=["True No", "True Yes"],
           title="Confusion matrix (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)


def save_feature_importance(results, pre, best_name):
    clf = results[best_name]["model"]
    if not hasattr(clf, "feature_importances_"):
        return
    cat_names = pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out()
    feat_names = list(NUMERIC) + list(cat_names)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(range(len(order)), imp[order][::-1], color="#22d3ee")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order[::-1]], fontsize=8)
    ax.set(title=f"Top 15 features — {best_name}", xlabel="importance")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=120)
    plt.close(fig)


def save_eda(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    churn = df[df[TARGET] == "Yes"]
    ax[0].bar(["Did not churn", "Churned"],
              [(df[TARGET] == "No").sum(), (df[TARGET] == "Yes").sum()], color="#22d3ee")
    ax[0].set(title="Class balance", ylabel="customers")
    churn_rate = df.groupby("Contract")[TARGET].apply(lambda s: (s == "Yes").mean() * 100)
    ax[1].bar(churn_rate.index, churn_rate.values, color="#6366f1")
    ax[1].set(title="Churn rate by contract", ylabel="% churned")
    ax[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"EDA hint: churn rate by contract -> {churn_rate.round(1).to_dict()}")


if __name__ == "__main__":
    df = load_data()
    results, pre, X_te, y_te = train_eval(df)

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best = results[best_name]

    # Save a single self-contained pipeline (preprocessor + best model)
    # so the demo only needs raw customer features as input.
    full_pipeline = Pipeline([
        ("pre", pre),
        ("clf", best["model"]),
    ])
    joblib_dump(full_pipeline, os.path.join(MODEL_DIR, "churn_model.joblib"))

    save_plots(results, best_name)
    save_feature_importance(results, pre, best_name)
    save_eda(df, os.path.join(MODEL_DIR, "churn_eda.png"))

    summary = {
        name: {k: round(v, 4) if isinstance(v, float) else v
               for k, v in r.items() if k not in ("model", "roc", "pr")}
        for name, r in results.items()
    }
    summary["best_model"] = best_name
    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Customer Churn results (20% stratified hold-out) ===")
    for name, r in summary.items():
        if isinstance(r, dict) and "roc_auc" in r:
            print(f"{name:14s} AUC={r['roc_auc']:.3f}  P={r['precision']:.3f}  "
                  f"R={r['recall']:.3f}  F1={r['f1']:.3f}")
    print(f"Best model: {summary['best_model']}")
    print(f"Artifacts saved to {MODEL_DIR}")