"""
04_train_heart_model.py
Full training pipeline for Task 4: Disease Prediction (Heart Disease, Cleveland subset).

Steps:
  1. Load data
  2. Clean (drop useless columns, binarize target)
  3. Preprocessing pipeline (impute + encode + scale), fit on train fold only
  4. Train/validation split (stratified)
  5. Train 3 models: Logistic Regression, Random Forest, HistGradientBoosting
  6. Hyperparameter tuning via RandomizedSearchCV (scored on ROC-AUC)
  7. Evaluate on held-out validation set (ROC-AUC, PR-AUC, precision/recall/F1)
  8. Pick the best model, retrain on full training data
  9. Save the fitted pipeline + metrics to disk
"""

import os
import json
import joblib
import numpy as np
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
data_path = "heart_disease_cleveland.csv"
out_dir = "outputs/heart_model"
os.makedirs(out_dir, exist_ok=True)

raw_target = "num"          # original column: 0 (healthy) .. 4 (severe)
target = "has_disease"      # binarized target we actually predict
random_state = 42
n_search_iter = 15
n_jobs = -1 if multiprocessing.cpu_count() > 1 else 1

numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
boolean_cols = ["fbs", "exang"]
categorical_cols = ["sex", "cp", "restecg", "slope", "thal"]


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(data_path)
    return df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # "id" is just a row identifier, "dataset" is constant ("Cleveland") once
    # filtered -- neither carries predictive signal.
    df = df.drop(columns=["id", "dataset"], errors="ignore")

    # Booleans -> 0/1 integers so every model can use them directly.
    for c in boolean_cols:
        df[c] = df[c].astype(int)

    # Binarize the target: 0 = no disease, 1-4 = some degree of disease present.
    # This matches the brief's framing ("predict the possibility of disease")
    # and lets us reuse the same Precision/Recall/F1/ROC-AUC toolkit as Task 1.
    df[target] = (df[raw_target] > 0).astype(int)
    df = df.drop(columns=[raw_target])

    return df


# ---------------------------------------------------------------------------
# 3. Preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # Booleans are already clean 0/1 ints with no missing values -- just pass through.
    boolean_transformer = "passthrough"

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("bool", boolean_transformer, boolean_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    return preprocessor


# ---------------------------------------------------------------------------
# 4. Model configurations
# ---------------------------------------------------------------------------
def get_model_configs():
    configs = {
        "logistic_regression": {
            "estimator": LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=random_state
            ),
            "param_distributions": {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced", random_state=random_state, n_jobs=n_jobs
            ),
            "param_distributions": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 7, None],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__max_features": ["sqrt", "log2"],
            },
        },
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingClassifier(random_state=random_state),
            "param_distributions": {
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [2, 3, 4, None],
                "model__max_iter": [100, 150, 200],
                "model__l2_regularization": [0.0, 0.1, 1.0],
                "model__class_weight": ["balanced"],
            },
        },
    }
    return configs


# ---------------------------------------------------------------------------
# 5. Train and tune
# ---------------------------------------------------------------------------
def train_and_tune(name, estimator, param_distributions, preprocessor, X_train, y_train, cv):
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", estimator),
    ])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_search_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
        error_score="raise",
    )
    print(f"\nTuning {name} ...")
    search.fit(X_train, y_train)
    print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(name, fitted_pipe, X_val, y_val, threshold=0.5):
    proba = fitted_pipe.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_val, preds)

    print(f"\n--- {name} (validation, threshold={threshold}) ---")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")

    return {
        "name": name, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "precision": precision, "recall": recall, "f1": f1, "proba": proba,
    }


def plot_roc_pr_curves(results, y_val):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for res in results:
        fpr, tpr, _ = roc_curve(y_val, res["proba"])
        axes[0].plot(fpr, tpr, label=f"{res['name']} (AUC={res['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_val, res["proba"])
        axes[1].plot(rec, prec, label=f"{res['name']} (AUC={res['pr_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend()

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_pr_curves.png"), dpi=120)
    plt.close(fig)


def get_feature_names(preprocessor):
    """Recover readable feature names after ColumnTransformer + OneHotEncoder."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    return numeric_cols + boolean_cols + cat_names


def plot_feature_importance(fitted_pipe, model_name):
    model = fitted_pipe.named_steps["model"]
    preprocessor = fitted_pipe.named_steps["preprocess"]
    feature_names = get_feature_names(preprocessor)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#C44E52",
    )
    ax.set_title(f"Feature Importance — {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"feature_importance_{model_name}.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    df = clean_data(df)
    print(f"Target balance:\n{df[target].value_counts(normalize=True)}")

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    configs = get_model_configs()
    fitted_models = {}
    cv_scores = {}
    for name, cfg in configs.items():
        best_pipe, best_cv_score = train_and_tune(
            name, cfg["estimator"], cfg["param_distributions"],
            preprocessor, X_train, y_train, cv,
        )
        fitted_models[name] = best_pipe
        cv_scores[name] = best_cv_score

    results = []
    for name, pipe in fitted_models.items():
        results.append(evaluate_model(name, pipe, X_val, y_val))

    plot_roc_pr_curves(results, y_val)

    best_result = max(results, key=lambda r: r["roc_auc"])
    best_name = best_result["name"]
    best_pipe = fitted_models[best_name]
    print(f"\n=== Best model: {best_name} (val ROC-AUC={best_result['roc_auc']:.4f}) ===")

    plot_feature_importance(best_pipe, best_name)

    print(f"\nRetraining {best_name} on full data ...")
    best_pipe.fit(X, y)

    model_path = os.path.join(out_dir, "heart_disease_pipeline.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"Model saved to: {model_path}")

    metadata = {
        "best_model": best_name,
        "feature_cols": feature_cols,
        "cv_roc_auc": cv_scores,
        "validation_metrics": {
            r["name"]: {
                "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
            } for r in results
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {os.path.join(out_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
