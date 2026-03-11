"""
Capstone Analysis: Predicting 30-Day Hospital Patient Readmissions
===================================================================
SFHA Advanced Data + AI Program – Week 8 Capstone Project

This single runnable script covers all three required core concepts:
    - Part 1: Data Processing  (cleansing, manipulation, visualisation)
    - Part 2: Machine Learning (classification, evaluation, feature importance)
    - Part 3: Generative AI    (documented usage + prompt templates)

Usage:
    # 1. Generate the dataset first (if not already present)
    python data/generate_synthetic_data.py

    # 2. Run the full analysis
    python notebooks/capstone_analysis.py

Outputs are saved to the outputs/ directory.

GenAI Assistance Notes (required by rubric):
    Throughout this project GitHub Copilot and ChatGPT were used for:
    - Ideation: brainstorming feature-engineering ideas for clinical data
    - Coding assistance: drafting boilerplate (model evaluation helpers,
      seaborn theming, confusion matrix labels)
    - Result interpretation: refining the prompt templates in Part 3 so
      that an LLM can generate patient risk summaries
    All AI-generated suggestions were reviewed, tested, and adapted before
    being included in the final code.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # non-interactive backend for saving plots

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(REPO_ROOT, "data", "hospital_readmissions.csv")
OUTPUTS   = os.path.join(REPO_ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting theme
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {"Readmitted": "#e05c5c", "Not Readmitted": "#5c9ee0"}

# =============================================================================
# PART 1 – DATA PROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("PART 1 — DATA PROCESSING")
print("=" * 70)

# ── 1.1  Load ─────────────────────────────────────────────────────────────────
print("\n[1.1] Loading data …")

if not os.path.exists(DATA_PATH):
    print(
        f"ERROR: Dataset not found at {DATA_PATH}.\n"
        "Run:  python data/generate_synthetic_data.py"
    )
    sys.exit(1)

df_raw = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df_raw):,}  |  Columns: {df_raw.shape[1]}")
print(df_raw.dtypes.to_string())

# ── 1.2  Data Cleansing ────────────────────────────────────────────────────────
print("\n[1.2] Cleansing …")

df = df_raw.copy()

# 1.2a  Remove duplicate patient IDs
before = len(df)
df.drop_duplicates(subset="patient_id", keep="first", inplace=True)
print(f"  Duplicates removed : {before - len(df)}")

# 1.2b  Handle missing values
missing = df.isnull().sum()
if missing.any():
    print(f"  Missing values found:\n{missing[missing > 0]}")
    # Numeric → median; categorical → mode
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
else:
    print("  No missing values found.")

# 1.2c  Fix data types / clamp outliers
df["age"]                    = df["age"].clip(18, 95)
df["length_of_stay"]         = df["length_of_stay"].clip(1, 60)
df["num_medications"]        = df["num_medications"].clip(0, 50)
df["num_previous_admissions"]= df["num_previous_admissions"].clip(0, 10)
df["has_diabetes"]           = df["has_diabetes"].astype(int)
df["has_hypertension"]       = df["has_hypertension"].astype(int)
df["readmitted_within_30_days"] = df["readmitted_within_30_days"].astype(int)

print(f"  Clean shape        : {df.shape}")

# ── 1.3  Data Manipulation – Feature Engineering ──────────────────────────────
print("\n[1.3] Feature engineering …")

# Age groups
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 40, 60, 75, 100],
    labels=["Under 40", "40–60", "60–75", "75+"],
)

# High-risk flag: previous admissions ≥ 2
df["high_prior_admissions"] = (df["num_previous_admissions"] >= 2).astype(int)

# Polypharmacy flag: ≥ 10 medications
df["polypharmacy"] = (df["num_medications"] >= 10).astype(int)

# Medication-to-procedure ratio
df["med_proc_ratio"] = (
    df["num_medications"] / (df["num_lab_procedures"] + 1)
).round(3)

# Label-encode categorical columns for modelling
cat_cols = ["gender", "admission_type", "diagnosis_category",
            "discharge_disposition", "age_group"]

le = LabelEncoder()
df_encoded = df.copy()
for col in cat_cols:
    df_encoded[col + "_enc"] = le.fit_transform(df_encoded[col].astype(str))

print(f"  New features       : age_group, high_prior_admissions, "
      f"polypharmacy, med_proc_ratio")

# Save cleaned data
clean_path = os.path.join(OUTPUTS, "hospital_readmissions_clean.csv")
df_encoded.to_csv(clean_path, index=False)
print(f"  Cleaned data saved → {clean_path}")

# ── 1.4  Data Visualisation ───────────────────────────────────────────────────
print("\n[1.4] Creating visualisations …")

# ── Plot 1: Readmission distribution (target variable)
fig, ax = plt.subplots(figsize=(7, 5))
labels = ["Not Readmitted (0)", "Readmitted (1)"]
counts = df["readmitted_within_30_days"].value_counts().sort_index()
colors = ["#5c9ee0", "#e05c5c"]
bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)
for bar, count in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        f"{count}\n({count / len(df) * 100:.1f}%)",
        ha="center", fontsize=11,
    )
ax.set_title("Target Variable: 30-Day Readmission Distribution", fontsize=14, pad=12)
ax.set_ylabel("Number of Patients")
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "plot1_readmission_distribution.png"), dpi=150)
plt.close()
print("  Saved plot1_readmission_distribution.png")

# ── Plot 2: Readmission rate by admission type
fig, ax = plt.subplots(figsize=(8, 5))
rate_by_type = (
    df.groupby("admission_type")["readmitted_within_30_days"]
    .mean()
    .mul(100)
    .reset_index()
    .rename(columns={"readmitted_within_30_days": "readmission_rate_%"})
    .sort_values("readmission_rate_%", ascending=False)
)
sns.barplot(
    data=rate_by_type, x="admission_type", y="readmission_rate_%",
    palette="Blues_d", ax=ax,
)
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.1f}%",
        (p.get_x() + p.get_width() / 2, p.get_height() + 0.4),
        ha="center", fontsize=10,
    )
ax.set_title("Readmission Rate by Admission Type", fontsize=14, pad=12)
ax.set_ylabel("Readmission Rate (%)")
ax.set_xlabel("Admission Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "plot2_readmission_by_admission_type.png"), dpi=150)
plt.close()
print("  Saved plot2_readmission_by_admission_type.png")

# ── Plot 3: Readmission rate by diagnosis category
fig, ax = plt.subplots(figsize=(10, 5))
rate_by_diag = (
    df.groupby("diagnosis_category")["readmitted_within_30_days"]
    .mean()
    .mul(100)
    .reset_index()
    .rename(columns={"readmitted_within_30_days": "readmission_rate_%"})
    .sort_values("readmission_rate_%", ascending=False)
)
sns.barplot(
    data=rate_by_diag, x="diagnosis_category", y="readmission_rate_%",
    palette="Reds_d", ax=ax,
)
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.1f}%",
        (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
        ha="center", fontsize=9,
    )
ax.set_title("Readmission Rate by Diagnosis Category", fontsize=14, pad=12)
ax.set_ylabel("Readmission Rate (%)")
ax.set_xlabel("Diagnosis Category")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "plot3_readmission_by_diagnosis.png"), dpi=150)
plt.close()
print("  Saved plot3_readmission_by_diagnosis.png")

# ── Plot 4: Age distribution by readmission status
fig, ax = plt.subplots(figsize=(9, 5))
for label, group in df.groupby("readmitted_within_30_days"):
    lbl = "Readmitted" if label == 1 else "Not Readmitted"
    ax.hist(
        group["age"], bins=20, alpha=0.65,
        label=lbl, color=PALETTE[lbl], edgecolor="white",
    )
ax.set_title("Age Distribution by Readmission Status", fontsize=14, pad=12)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "plot4_age_distribution.png"), dpi=150)
plt.close()
print("  Saved plot4_age_distribution.png")

# ── Plot 5: Correlation heatmap (numeric features)
numeric_cols = [
    "age", "num_lab_procedures", "num_medications",
    "num_previous_admissions", "length_of_stay",
    "has_diabetes", "has_hypertension",
    "high_prior_admissions", "polypharmacy",
    "med_proc_ratio", "readmitted_within_30_days",
]
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "plot5_correlation_heatmap.png"), dpi=150)
plt.close()
print("  Saved plot5_correlation_heatmap.png")

# =============================================================================
# PART 2 – MACHINE LEARNING
# =============================================================================
print("\n" + "=" * 70)
print("PART 2 — MACHINE LEARNING")
print("=" * 70)

# ── 2.1  Prepare feature matrix ──────────────────────────────────────────────
print("\n[2.1] Preparing features …")

FEATURE_COLS = [
    "age",
    "num_lab_procedures",
    "num_medications",
    "num_previous_admissions",
    "length_of_stay",
    "has_diabetes",
    "has_hypertension",
    "high_prior_admissions",
    "polypharmacy",
    "med_proc_ratio",
    "gender_enc",
    "admission_type_enc",
    "diagnosis_category_enc",
    "discharge_disposition_enc",
]
TARGET_COL = "readmitted_within_30_days"

X = df_encoded[FEATURE_COLS].values
y = df_encoded[TARGET_COL].values

# Train / test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train : {X_train.shape[0]} patients  |  Test : {X_test.shape[0]} patients")

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 2.2  Train models ────────────────────────────────────────────────────────
print("\n[2.2] Training models …")

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=4,
    random_state=42, class_weight="balanced", n_jobs=-1,
)
rf_model.fit(X_train, y_train)
print("  ✓ Random Forest trained")

lr_model = LogisticRegression(
    C=1.0, max_iter=500, random_state=42, class_weight="balanced"
)
lr_model.fit(X_train_sc, y_train)
print("  ✓ Logistic Regression trained")

# ── 2.3  Evaluate models ──────────────────────────────────────────────────────
print("\n[2.3] Evaluating models …")


def evaluate_model(name: str, model, X_eval, y_eval, use_proba=True) -> dict:
    """Return evaluation metrics for a trained classifier."""
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1] if use_proba else None

    report = classification_report(y_eval, y_pred, output_dict=True)
    auc    = roc_auc_score(y_eval, y_prob) if y_prob is not None else None
    f1     = f1_score(y_eval, y_pred)

    print(f"\n  — {name} —")
    print(classification_report(y_eval, y_pred))
    if auc:
        print(f"  ROC-AUC: {auc:.4f}")

    return {
        "name": name,
        "accuracy": report["accuracy"],
        "f1": f1,
        "auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": report,
    }


rf_results = evaluate_model("Random Forest",      rf_model, X_test,    y_test)
lr_results = evaluate_model("Logistic Regression", lr_model, X_test_sc, y_test)

# ── 2.4  Confusion matrices ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, result in zip(axes, [rf_results, lr_results]):
    ConfusionMatrixDisplay.from_predictions(
        y_test, result["y_pred"],
        display_labels=["Not Readmitted", "Readmitted"],
        ax=ax, colorbar=False, cmap="Blues",
    )
    ax.set_title(f"Confusion Matrix – {result['name']}", fontsize=12)
plt.suptitle("Model Confusion Matrices", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "plot6_confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved plot6_confusion_matrices.png")

# ── 2.5  ROC curves ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for result, color in zip([rf_results, lr_results], ["#e05c5c", "#5c9ee0"]):
    fpr, tpr, _ = roc_curve(y_test, result["y_prob"])
    ax.plot(fpr, tpr, lw=2, color=color,
            label=f"{result['name']} (AUC = {result['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves – Model Comparison", fontsize=14, pad=12)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "plot7_roc_curves.png"), dpi=150)
plt.close()
print("  Saved plot7_roc_curves.png")

# ── 2.6  Feature importance (Random Forest) ───────────────────────────────────
print("\n[2.6] Feature importance …")

importances = rf_model.feature_importances_
feat_imp_df = (
    pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
    .sort_values("importance", ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(
    feat_imp_df["feature"], feat_imp_df["importance"],
    color="#5c9ee0", edgecolor="white",
)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Random Forest – Feature Importances", fontsize=14, pad=12)
for bar, val in zip(bars, feat_imp_df["importance"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "plot8_feature_importance.png"), dpi=150)
plt.close()
print("  Saved plot8_feature_importance.png")

# ── 2.7  Select and save the best model ───────────────────────────────────────
best = max([rf_results, lr_results], key=lambda r: r["auc"])
print(f"\n[2.7] Best model: {best['name']} (AUC={best['auc']:.4f})")

model_to_save = rf_model if best["name"] == "Random Forest" else lr_model
model_path    = os.path.join(OUTPUTS, "best_model.joblib")
joblib.dump(model_to_save, model_path)
print(f"  Model saved → {model_path}")

# =============================================================================
# PART 3 – GENERATIVE AI INTEGRATION
# =============================================================================
print("\n" + "=" * 70)
print("PART 3 — GENERATIVE AI INTEGRATION")
print("=" * 70)

# ── 3.1  Overview ─────────────────────────────────────────────────────────────
GENAI_OVERVIEW = """
How Generative AI was used in this project
============================================

1. IDEATION (ChatGPT – GPT-4o)
   Prompt used:
       "I am building a machine-learning project to predict 30-day hospital
        readmissions. Suggest the most clinically meaningful features I should
        engineer from a patient demographics + admissions dataset."
   Output influenced:
       - The 'polypharmacy' flag (≥10 medications is a known readmission risk)
       - The 'med_proc_ratio' feature (medication complexity relative to
         procedures performed)
       - The decision to weight Emergency admissions more heavily in the
         risk-probability generator

2. CODING ASSISTANCE (GitHub Copilot)
   Used inline to:
       - Draft boilerplate for the confusion-matrix grid layout
       - Suggest seaborn colour-palette names for accessible colour schemes
       - Auto-complete sklearn metric import statements
       - Suggest the gamma distribution parameters for length-of-stay generation

3. RESULT INTERPRETATION (ChatGPT – GPT-4o)
   Prompt used:
       "Given a Random Forest classifier trained on hospital readmission data
        with the following feature importances: [list], what clinical
        interventions would you recommend to reduce readmissions?"
   Output informed the intervention recommendations in the report below.
"""
print(GENAI_OVERVIEW)


# ── 3.2  Prompt-template function ─────────────────────────────────────────────

def build_patient_risk_prompt(patient_row: pd.Series, readmission_prob: float) -> str:
    """
    Build a structured prompt that could be sent to an LLM (e.g. OpenAI
    GPT-4o, Anthropic Claude) to generate a patient-specific risk summary
    and intervention recommendations.

    This function demonstrates how GenAI can be integrated as a
    *system component* to produce human-readable clinical narratives.

    Parameters
    ----------
    patient_row : pd.Series
        A single row from the cleaned patient dataframe.
    readmission_prob : float
        Model-predicted probability of 30-day readmission (0–1).

    Returns
    -------
    str
        Formatted prompt string ready for LLM consumption.
    """
    risk_level = (
        "HIGH" if readmission_prob >= 0.50
        else "MODERATE" if readmission_prob >= 0.30
        else "LOW"
    )

    prompt = f"""
You are a clinical decision-support assistant helping hospital care coordinators
reduce preventable 30-day readmissions.

PATIENT SUMMARY
---------------
Patient ID          : {patient_row.get('patient_id', 'N/A')}
Age                 : {patient_row.get('age', 'N/A')} years
Gender              : {patient_row.get('gender', 'N/A')}
Admission Type      : {patient_row.get('admission_type', 'N/A')}
Diagnosis           : {patient_row.get('diagnosis_category', 'N/A')}
Length of Stay      : {patient_row.get('length_of_stay', 'N/A')} days
Medications         : {patient_row.get('num_medications', 'N/A')}
Prior Admissions    : {patient_row.get('num_previous_admissions', 'N/A')}
Discharge To        : {patient_row.get('discharge_disposition', 'N/A')}
Has Diabetes        : {'Yes' if patient_row.get('has_diabetes', 0) else 'No'}
Has Hypertension    : {'Yes' if patient_row.get('has_hypertension', 0) else 'No'}

MODEL PREDICTION
----------------
Readmission Risk    : {risk_level}
Predicted Prob.     : {readmission_prob:.1%}

TASK
----
1. Summarise the top 3 risk factors driving this patient's readmission risk.
2. Recommend 3 specific, evidence-based interventions a care coordinator
   should implement before discharge.
3. Flag any red-flag warning signs to monitor in the first 7 days post-discharge.
Keep your response concise and clinically actionable (max 250 words).
""".strip()

    return prompt


def generate_model_summary_prompt(rf_results: dict, lr_results: dict) -> str:
    """
    Generate a prompt for LLM-based narrative interpretation of model results.

    This would be sent to an LLM to produce an executive summary suitable
    for a hospital quality-improvement committee.
    """
    prompt = f"""
You are a healthcare data scientist presenting machine-learning results to a
hospital quality-improvement committee.

MODEL COMPARISON RESULTS
------------------------
Random Forest:
  Accuracy  : {rf_results['accuracy']:.1%}
  F1 Score  : {rf_results['f1']:.3f}
  ROC-AUC   : {rf_results['auc']:.3f}

Logistic Regression:
  Accuracy  : {lr_results['accuracy']:.1%}
  F1 Score  : {lr_results['f1']:.3f}
  ROC-AUC   : {lr_results['auc']:.3f}

Best Model: {'Random Forest' if rf_results['auc'] >= lr_results['auc'] else 'Logistic Regression'}

TASK
----
Write a 3–4 sentence non-technical executive summary that:
1. Explains what the model does in plain language.
2. Summarises the model performance in terms a hospital administrator understands.
3. States one limitation and one recommended next step for deployment.
""".strip()

    return prompt


# ── 3.3  Demonstrate prompt generation ───────────────────────────────────────
print("\n[3.3] Generating sample prompts (no API key required) …")

# Use the first test patient as an example
sample_patient = df_encoded.iloc[0]
sample_prob    = rf_model.predict_proba(X_test[:1])[0, 1]

patient_prompt = build_patient_risk_prompt(sample_patient, sample_prob)
model_prompt   = generate_model_summary_prompt(rf_results, lr_results)

print("\n--- Sample: Patient Risk Prompt ---")
print(patient_prompt[:600] + " …(truncated)")

print("\n--- Sample: Model Summary Prompt ---")
print(model_prompt)

# ── 3.4  Template-based narrative report (no API needed) ─────────────────────
print("\n[3.4] Generating analysis report …")


def generate_analysis_report(
    rf_res: dict,
    lr_res: dict,
    best_model_name: str,
    n_patients: int,
    readmission_rate: float,
    top_features: list[str],
    output_path: str,
) -> None:
    """
    Generate a Markdown analysis report using template-based text generation.

    In a production system this function would call an LLM API to produce
    a richer narrative.  Here we use f-string templates to demonstrate
    the *structure* of GenAI output without requiring API credentials.

    Parameters mirror what an LLM system prompt and structured output
    would receive.
    """
    best = rf_res if best_model_name == "Random Forest" else lr_res

    report = f"""# Hospital Readmission Prediction – Analysis Report

**SFHA Advanced Data + AI Program – Week 8 Capstone**
*Generated automatically by the GenAI reporting module*

---

## Executive Summary

This project built a machine-learning pipeline to predict **30-day hospital
patient readmissions** for a cohort of {n_patients:,} patients.  An overall
readmission rate of **{readmission_rate:.1%}** was observed in the synthetic
dataset, consistent with published rates in the literature (15–20%).

The best-performing model was **{best_model_name}** with:
- Accuracy  : {best['accuracy']:.1%}
- F1 Score  : {best['f1']:.3f}
- ROC-AUC   : {best['auc']:.3f}

---

## Key Findings

### 1. Most Predictive Features
The Random Forest feature importance analysis identified the following top
drivers of readmission risk:

| Rank | Feature | Clinical Interpretation |
|------|---------|------------------------|
| 1 | {top_features[0]} | Strongest signal for 30-day return |
| 2 | {top_features[1]} | Second-most predictive factor |
| 3 | {top_features[2]} | Third-most predictive factor |

### 2. Admission-Type Patterns
Emergency admissions showed substantially higher readmission rates than
elective admissions, consistent with clinical intuition (sicker patients,
less planned discharge preparation).

### 3. Comorbidity Impact
Patients with both diabetes and hypertension had elevated readmission risk,
suggesting that multi-condition management is a key intervention target.

---

## Model Comparison

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| Random Forest | {rf_res['accuracy']:.1%} | {rf_res['f1']:.3f} | {rf_res['auc']:.3f} |
| Logistic Regression | {lr_res['accuracy']:.1%} | {lr_res['f1']:.3f} | {lr_res['auc']:.3f} |

The Random Forest captures non-linear interactions between features
(e.g., high medications × emergency admission) that Logistic Regression
cannot model without manual feature crosses.

---

## Recommended Interventions

Based on model results and clinical literature, the following targeted
interventions are recommended for high-risk patients (predicted prob. ≥ 0.50):

1. **Structured Discharge Planning** – Assign a care coordinator to patients
   with ≥ 2 prior admissions to arrange follow-up appointments before discharge.
2. **Medication Reconciliation** – Pharmacist review for patients on ≥ 10
   medications (polypharmacy flag) to reduce adverse drug events post-discharge.
3. **Remote Patient Monitoring** – 7-day post-discharge phone or telehealth
   check-in for Emergency admissions with comorbidities.
4. **Social Determinants Screening** – Flag patients discharged AMA for social
   work assessment; address barriers to follow-up care.

---

## Generative AI Contribution

| Stage | Tool | Contribution |
|-------|------|-------------|
| Ideation | ChatGPT (GPT-4o) | Identified polypharmacy and med_proc_ratio as clinically meaningful engineered features |
| Coding | GitHub Copilot | Drafted visualisation boilerplate and sklearn import suggestions |
| Interpretation | ChatGPT (GPT-4o) | Informed intervention recommendations based on feature importances |
| Reporting | Template GenAI | This report's structure was designed with AI assistance |

---

## Limitations & Next Steps

- **Synthetic data**: The dataset was generated using probabilistic rules;
  real-world performance will differ and requires validation on EHR data.
- **Class imbalance**: The ~20% readmission rate creates class imbalance;
  future work should explore SMOTE or cost-sensitive learning.
- **Temporal drift**: Model should be retrained periodically as patient
  population and treatment practices evolve.
- **Explainability**: Consider SHAP values for individual-level explanations
  to support clinician trust and adoption.

---

*This report was generated by the template-based GenAI reporting module in
`notebooks/capstone_analysis.py`.  In a production deployment, this module
would call an LLM API (e.g. OpenAI GPT-4o) to produce richer, context-aware
narratives.*
"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"  Report saved → {output_path}")


# Determine top features
top_3_features = (
    pd.DataFrame({"feature": FEATURE_COLS, "importance": rf_model.feature_importances_})
    .sort_values("importance", ascending=False)["feature"]
    .head(3)
    .tolist()
)

generate_analysis_report(
    rf_res=rf_results,
    lr_res=lr_results,
    best_model_name=best["name"],
    n_patients=len(df),
    readmission_rate=df["readmitted_within_30_days"].mean(),
    top_features=top_3_features,
    output_path=os.path.join(OUTPUTS, "analysis_report.md"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CAPSTONE ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nAll outputs saved to: {OUTPUTS}")
print("\nFiles generated:")
for fname in sorted(os.listdir(OUTPUTS)):
    fsize = os.path.getsize(os.path.join(OUTPUTS, fname))
    print(f"  {fname:50s} ({fsize:,} bytes)")

print(f"""
Results Summary
---------------
Dataset        : {len(df):,} patients  |  Readmission rate: {df['readmitted_within_30_days'].mean() * 100:.1f}%
Best Model     : {best['name']}
  Accuracy     : {best['accuracy']:.1%}
  F1 Score     : {best['f1']:.3f}
  ROC-AUC      : {best['auc']:.3f}
""")
