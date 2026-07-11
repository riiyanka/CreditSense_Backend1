"""
Train the core 3-model credit risk cascade:
  Model 1: Default Probability Predictor  (binary classifier)
  Model 2: Loan Grade Classifier          (5-class: A-E)
  Model 3: Interest Rate Predictor        (regressor)

Cascade design: Model 2 uses Model 1's (out-of-fold) default-probability as an
extra feature; Model 3 uses both Model 1's and Model 2's OOF outputs. This
mirrors how the models will actually be called in production, one after the
other, and avoids the target leakage you'd get from naively using in-sample
predictions as features.

Run:  python train.py
Produces artifacts/*.joblib + artifacts/metrics.json
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, mean_absolute_error, r2_score, classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 3

# ---------------------------------------------------------------------------
# 1. Feature definition
# ---------------------------------------------------------------------------
# Excluded on purpose:
#   - IDs / PII: applicant_id, full_name, application_date, pan_number,
#     phone_number, address_id, device_id
#   - Fraud-graph ground truth (separate module, not this cascade):
#     fraud_ring_id, is_fraud_ring_member
#   - Uplift-experiment columns (not known at scoring time / different module):
#     rate_perturbation_pp, rate_offered_pct, loan_accepted
#   - Circular / leaky derived fields (computed FROM the assigned interest
#     rate, so using them as inputs would leak the answer):
#     proposed_emi, foir_total
#   - Downstream derived outputs, not model inputs:
#     credit_risk_score, expected_roi_pct

CATEGORICAL_FEATURES = [
    "state", "city_tier", "rural_urban", "gender",
    "employment_type", "sector", "loan_purpose",
]

NUMERIC_FEATURES = [
    "age", "years_at_current_job", "monthly_income", "num_dependents",
    "loan_amount_requested", "loan_tenure_months", "existing_emi",
    "foir_existing", "loan_to_income_ratio", "disposable_income",
    "cibil_score", "credit_history_months", "num_active_loans",
    "credit_utilization", "num_enquiries_6m", "dpd_max_12m",
    "loan_stacking_count_90d", "bounce_count_12m", "avg_bank_balance",
    "upi_txn_freq_monthly", "upi_txn_volatility", "gst_compliance_score",
    "sector_risk_score",
]

BASE_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
GRADE_ORDER = ["A", "B", "C", "D", "E"]


def build_preprocessor(categorical, numeric):
    """OneHot for categoricals, median-impute + missingness flag for numerics.
    cibil_score (~19% missing = new-to-credit) and gst_compliance_score
    (~65% missing = not self-employed/business) have DESIGNED missingness,
    so add_indicator=True keeps that signal instead of throwing it away."""
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


def main():
    print("Loading historical dataset...")
    df = pd.read_csv(DATA_DIR / "historical_dataset.csv")
    print(f"  {len(df):,} rows")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["loan_grade"]
    )
    print(f"  train: {len(train_df):,}  |  internal validation: {len(val_df):,}")

    X_train_base = train_df[BASE_FEATURES]
    X_val_base = val_df[BASE_FEATURES]

    metrics = {}

    # =======================================================================
    # MODEL 1: Default Probability Predictor
    # =======================================================================
    print("\n=== Model 1: Default Probability Predictor ===")
    y_train_default = train_df["default_flag"]
    y_val_default = val_df["default_flag"]

    pre1 = build_preprocessor(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    model1 = Pipeline([
        ("pre", pre1),
        ("clf", HistGradientBoostingClassifier(
            max_iter=150, max_depth=6, learning_rate=0.08,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])

    # Out-of-fold default-probability predictions on the TRAIN split only,
    # used as a feature for Model 2/3 without leaking the final Model 1 fit.
    print("  Computing 5-fold OOF predictions for stacking...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_default_proba = cross_val_predict(
        model1, X_train_base, y_train_default, cv=skf, method="predict_proba", n_jobs=-1
    )[:, 1]

    print("  Fitting final Model 1 on full training split...")
    model1.fit(X_train_base, y_train_default)
    val_default_proba = model1.predict_proba(X_val_base)[:, 1]

    m1_auc = roc_auc_score(y_val_default, val_default_proba)
    m1_ap = average_precision_score(y_val_default, val_default_proba)
    print(f"  Validation ROC-AUC: {m1_auc:.4f}  |  PR-AUC: {m1_ap:.4f}")
    metrics["model1_default_predictor"] = {"roc_auc": m1_auc, "pr_auc": m1_ap}

    # =======================================================================
    # MODEL 2: Loan Grade Classifier (uses Model 1's OOF proba as a feature)
    # =======================================================================
    print("\n=== Model 2: Loan Grade Classifier ===")
    X_train_m2 = X_train_base.copy()
    X_train_m2["default_proba"] = oof_default_proba
    X_val_m2 = X_val_base.copy()
    X_val_m2["default_proba"] = val_default_proba

    y_train_grade = train_df["loan_grade"]
    y_val_grade = val_df["loan_grade"]

    m2_numeric = NUMERIC_FEATURES + ["default_proba"]
    pre2 = build_preprocessor(CATEGORICAL_FEATURES, m2_numeric)
    model2 = Pipeline([
        ("pre", pre2),
        ("clf", HistGradientBoostingClassifier(
            max_iter=150, max_depth=6, learning_rate=0.08,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])

    print("  Computing 5-fold OOF predictions for stacking...")
    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_grade_proba = cross_val_predict(
        model2, X_train_m2, y_train_grade, cv=skf2, method="predict_proba", n_jobs=-1
    )
    # class order from the estimator fitted on full data (fit once to get classes_)
    model2.fit(X_train_m2, y_train_grade)
    grade_classes = list(model2.named_steps["clf"].classes_)

    val_grade_pred = model2.predict(X_val_m2)
    val_grade_proba = model2.predict_proba(X_val_m2)

    m2_acc = accuracy_score(y_val_grade, val_grade_pred)
    m2_f1 = f1_score(y_val_grade, val_grade_pred, average="macro")
    print(f"  Validation accuracy: {m2_acc:.4f}  |  macro-F1: {m2_f1:.4f}")
    print(classification_report(y_val_grade, val_grade_pred))
    metrics["model2_grade_classifier"] = {
        "accuracy": m2_acc, "macro_f1": m2_f1, "classes": grade_classes
    }

    # =======================================================================
    # MODEL 3: Interest Rate Predictor (uses Model 1 + Model 2 OOF outputs)
    # =======================================================================
    print("\n=== Model 3: Interest Rate Predictor ===")
    oof_grade_df = pd.DataFrame(
        oof_grade_proba, columns=[f"grade_proba_{c}" for c in grade_classes],
        index=X_train_m2.index,
    )
    val_grade_df = pd.DataFrame(
        val_grade_proba, columns=[f"grade_proba_{c}" for c in grade_classes],
        index=X_val_m2.index,
    )

    X_train_m3 = pd.concat([X_train_m2, oof_grade_df], axis=1)
    X_val_m3 = pd.concat([X_val_m2, val_grade_df], axis=1)

    y_train_rate = train_df["interest_rate_pct"]
    y_val_rate = val_df["interest_rate_pct"]

    m3_numeric = NUMERIC_FEATURES + ["default_proba"] + list(oof_grade_df.columns)
    pre3 = build_preprocessor(CATEGORICAL_FEATURES, m3_numeric)
    model3 = Pipeline([
        ("pre", pre3),
        ("reg", HistGradientBoostingRegressor(
            max_iter=150, max_depth=6, learning_rate=0.08,
            random_state=RANDOM_STATE,
        )),
    ])
    model3.fit(X_train_m3, y_train_rate)
    val_rate_pred = model3.predict(X_val_m3)

    m3_mae = mean_absolute_error(y_val_rate, val_rate_pred)
    m3_r2 = r2_score(y_val_rate, val_rate_pred)
    print(f"  Validation MAE: {m3_mae:.3f} pp  |  R2: {m3_r2:.4f}")
    metrics["model3_rate_predictor"] = {"mae_pp": m3_mae, "r2": m3_r2}

    # =======================================================================
    # Out-of-time check on current_dataset.csv (drift sanity check)
    # =======================================================================
    print("\n=== Out-of-time check on current_dataset.csv ===")
    cur = pd.read_csv(DATA_DIR / "current_dataset.csv")
    Xc_base = cur[BASE_FEATURES]
    cur_default_proba = model1.predict_proba(Xc_base)[:, 1]
    cur_auc = roc_auc_score(cur["default_flag"], cur_default_proba)
    print(f"  Model 1 ROC-AUC on current (out-of-time) data: {cur_auc:.4f}")

    Xc_m2 = Xc_base.copy()
    Xc_m2["default_proba"] = cur_default_proba
    cur_grade_pred = model2.predict(Xc_m2)
    cur_grade_proba = model2.predict_proba(Xc_m2)
    cur_acc = accuracy_score(cur["loan_grade"], cur_grade_pred)
    print(f"  Model 2 accuracy on current (out-of-time) data: {cur_acc:.4f}")

    cur_grade_df = pd.DataFrame(
        cur_grade_proba, columns=[f"grade_proba_{c}" for c in grade_classes], index=Xc_m2.index
    )
    Xc_m3 = pd.concat([Xc_m2, cur_grade_df], axis=1)
    cur_rate_pred = model3.predict(Xc_m3)
    cur_mae = mean_absolute_error(cur["interest_rate_pct"], cur_rate_pred)
    print(f"  Model 3 MAE on current (out-of-time) data: {cur_mae:.3f} pp")
    print("  (Some drop vs. validation is expected -- current_dataset.csv has")
    print("   real drift: wage inflation, +0.75pp rate pass-through, higher gig %.)")

    metrics["out_of_time_current_dataset"] = {
        "model1_roc_auc": cur_auc, "model2_accuracy": cur_acc, "model3_mae_pp": cur_mae,
    }

    # =======================================================================
    # Save artifacts
    # =======================================================================
    joblib.dump(model1, ARTIFACT_DIR / "model1_default.joblib")
    joblib.dump(model2, ARTIFACT_DIR / "model2_grade.joblib")
    joblib.dump(model3, ARTIFACT_DIR / "model3_rate.joblib")
    joblib.dump(grade_classes, ARTIFACT_DIR / "grade_classes.joblib")

    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved models + metrics.json to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
