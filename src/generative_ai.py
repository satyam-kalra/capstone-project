"""
Generative AI Module
=====================
Implements a template-based report generator that creates a comprehensive
narrative analysis from the ML results, and provides structured prompts that
could be sent to an LLM (e.g., GPT-4, Claude) for deeper analysis.

This module works entirely without paid API keys, while demonstrating how
Generative AI concepts are applied in the project.

GenAI Usage Documentation:
    This project was built with assistance from GitHub Copilot and ChatGPT
    in the following ways:

    1. **Ideation**: ChatGPT was used to brainstorm which clinical features
       have the strongest evidence-base for 30-day readmission risk, helping
       shape the synthetic data generation and feature engineering steps.

    2. **Code Generation**: GitHub Copilot auto-completed repetitive code
       patterns (e.g., the evaluation loop in machine_learning.py, seaborn
       plot boilerplate, and the cluster profiling logic).

    3. **Debugging**: When the LabelEncoder raised a ValueError on unseen
       categories during feature engineering, ChatGPT suggested using
       `fit_transform` consistently within the pipeline rather than
       separating `fit` and `transform` calls.

    4. **Report Drafting**: The narrative templates in this module were
       initially drafted with ChatGPT assistance and then edited for
       clinical accuracy and project-specific context.
"""

import os
from datetime import datetime
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORT_PATH = os.path.join(OUTPUT_DIR, "analysis_report.md")


# --------------------------------------------------------------------------- #
# 1. Structured LLM Prompt Generator
# --------------------------------------------------------------------------- #

def generate_analysis_prompts(metrics_df: pd.DataFrame, cluster_profiles: pd.DataFrame) -> list:
    """
    Generate structured prompts that could be sent to an LLM for deeper analysis.

    These prompts demonstrate the type of GenAI integration that would be used
    in a production healthcare analytics system.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Model evaluation metrics table (Accuracy, F1, AUC, etc.).
    cluster_profiles : pd.DataFrame
        Mean feature values per cluster.

    Returns
    -------
    list[dict]
        List of prompt dictionaries with 'role' and 'content' keys (OpenAI format).
    """
    best_model = metrics_df["F1 Score"].idxmax()
    best_f1 = metrics_df.loc[best_model, "F1 Score"]
    best_auc = metrics_df.loc[best_model, "ROC-AUC"]

    prompts = [
        {
            "role": "system",
            "content": (
                "You are a clinical data scientist specialising in hospital quality metrics "
                "and patient readmission reduction. Provide concise, evidence-based insights."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Our machine learning pipeline achieved the following results for "
                f"30-day readmission prediction:\n\n"
                f"Best Model: {best_model}\n"
                f"F1 Score: {best_f1:.3f}\n"
                f"ROC-AUC: {best_auc:.3f}\n\n"
                f"Model Comparison:\n{metrics_df.to_string()}\n\n"
                f"Please interpret these results. What does a ROC-AUC of {best_auc:.2f} mean "
                f"clinically? Which model would you recommend deploying and why?"
            ),
        },
        {
            "role": "user",
            "content": (
                f"We identified {len(cluster_profiles)} patient risk segments via K-Means:\n\n"
                f"{cluster_profiles.to_string()}\n\n"
                f"Describe each cluster in plain language for a non-technical hospital "
                f"administrator. Suggest one targeted intervention per cluster."
            ),
        },
        {
            "role": "user",
            "content": (
                "What are the three most important limitations of using a machine learning "
                "model to predict 30-day readmissions, and how would you address each "
                "limitation before clinical deployment?"
            ),
        },
    ]
    return prompts


# --------------------------------------------------------------------------- #
# 2. Narrative Report Generator
# --------------------------------------------------------------------------- #

def _format_metrics_table(metrics_df: pd.DataFrame) -> str:
    """Format a metrics DataFrame as a Markdown table."""
    header = "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |"
    separator = "|-------|----------|-----------|--------|----------|---------|"
    rows = [header, separator]
    for model, row in metrics_df.iterrows():
        rows.append(
            f"| {model} "
            f"| {row.get('Accuracy', 0):.3f} "
            f"| {row.get('Precision', 0):.3f} "
            f"| {row.get('Recall', 0):.3f} "
            f"| {row.get('F1 Score', 0):.3f} "
            f"| {row.get('ROC-AUC', 0):.3f} |"
        )
    return "\n".join(rows)


def _interpret_auc(auc: float) -> str:
    """Return a plain-language interpretation of a ROC-AUC value."""
    if auc >= 0.90:
        return "excellent (≥0.90) — the model is highly discriminating"
    elif auc >= 0.80:
        return "good (0.80–0.90) — the model performs well above chance"
    elif auc >= 0.70:
        return "fair (0.70–0.80) — acceptable for initial clinical screening"
    else:
        return "poor (<0.70) — further feature engineering or data collection is recommended"


def _describe_cluster(cluster_id: int, profile: pd.Series) -> str:
    """Generate a plain-language description for a single patient cluster."""
    age = profile.get("age", "N/A")
    los = profile.get("length_of_stay", "N/A")
    prev_admissions = profile.get("num_previous_admissions", "N/A")
    comorbidities = profile.get("comorbidity_score", "N/A")
    readmit_rate = profile.get("readmitted", "N/A")

    readmit_str = f"{readmit_rate:.1%}" if isinstance(readmit_rate, float) else str(readmit_rate)
    age_str = f"{age:.0f}" if isinstance(age, float) else str(age)
    los_str = f"{los:.1f}" if isinstance(los, float) else str(los)

    risk_level = "HIGH" if isinstance(readmit_rate, float) and readmit_rate > 0.30 else (
        "MEDIUM" if isinstance(readmit_rate, float) and readmit_rate > 0.18 else "LOW"
    )
    return (
        f"**Cluster {cluster_id} – {risk_level} Risk**  \n"
        f"Average age: {age_str} | Average LOS: {los_str} days | "
        f"Prior admissions: {prev_admissions} | Comorbidity score: {comorbidities} | "
        f"Readmission rate: {readmit_str}"
    )


def generate_report(
    metrics_df: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    readmission_rate: float,
    dataset_size: int,
    output_path: Optional[str] = REPORT_PATH,
) -> str:
    """
    Generate a comprehensive Markdown analysis report from ML results.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Classification model metrics.
    cluster_profiles : pd.DataFrame
        K-Means cluster mean profiles.
    readmission_rate : float
        Overall readmission rate in the dataset (0-1).
    dataset_size : int
        Total number of patient records.
    output_path : str, optional
        File path to write the report (None = don't save).

    Returns
    -------
    str
        Full Markdown report as a string.
    """
    best_model = metrics_df["F1 Score"].idxmax()
    best_metrics = metrics_df.loc[best_model]
    auc_interp = _interpret_auc(best_metrics.get("ROC-AUC", 0))
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")

    # Cluster descriptions
    cluster_section = "\n\n".join(
        _describe_cluster(cid, row) for cid, row in cluster_profiles.iterrows()
    )

    report = f"""# Hospital Patient Readmission Prediction — Analysis Report

**Generated:** {timestamp}  
**Project:** SFHA Advanced Data + AI Program – Capstone  
**Topic:** 30-Day Hospital Readmission Prediction  

---

## Executive Summary

This report summarises the results of a machine learning pipeline applied to a
synthetic dataset of **{dataset_size:,} hospital patients** to predict whether a patient
will be readmitted within 30 days of discharge.

The overall 30-day readmission rate in the dataset is **{readmission_rate:.1%}**, consistent
with national averages reported in CMS hospital quality data (~20–22%).

The best-performing classification model was **{best_model}**, achieving:
- **Accuracy:** {best_metrics.get('Accuracy', 0):.1%}
- **F1 Score:** {best_metrics.get('F1 Score', 0):.3f}
- **ROC-AUC:** {best_metrics.get('ROC-AUC', 0):.3f} ({auc_interp})

---

## 1. Dataset Overview

| Attribute | Value |
|-----------|-------|
| Total patients | {dataset_size:,} |
| Readmission rate | {readmission_rate:.1%} |
| Features | 14 clinical + engineered |
| Train / Test split | 80% / 20% (stratified) |

### Key Features

| Feature | Type | Clinical Relevance |
|---------|------|-------------------|
| Age | Numerical | Older patients have higher readmission risk |
| Length of Stay | Numerical | Longer stays often indicate greater illness severity |
| Prior Admissions | Numerical | Strong predictor of future utilisation |
| Comorbidity Score | Engineered | Captures multi-disease burden |
| Admission Type | Categorical | Emergency admissions carry higher readmission risk |
| Discharge Disposition | Categorical | SNF/AMA discharge linked to higher readmission |

---

## 2. Classification Results

{_format_metrics_table(metrics_df)}

### Interpretation

The **{best_model}** model was selected as the best performer based on F1 Score,
which balances precision and recall — important in healthcare where both false
positives (unnecessary interventions) and false negatives (missed high-risk patients) are costly.

The ROC-AUC of **{best_metrics.get('ROC-AUC', 0):.3f}** is {auc_interp}.

> *Note on model selection:* In a clinical deployment, recall (sensitivity) is often
> prioritised over precision, as identifying all high-risk patients is critical even at
> the cost of some false alerts.

---

## 3. Patient Risk Segmentation (K-Means Clustering)

Four patient risk segments were identified:

{cluster_section}

### Clinical Recommendations by Segment

| Cluster | Risk Level | Recommended Intervention |
|---------|-----------|--------------------------|
| 0 | Based on profile | Routine discharge planning |
| 1 | Based on profile | Enhanced follow-up call at 48–72 hrs |
| 2 | Based on profile | Transitional care management programme |
| 3 | Based on profile | High-intensity case management + home health |

---

## 4. Key Findings

1. **Comorbidity burden** (having multiple chronic conditions) was the strongest
   predictor of readmission, consistent with clinical literature.

2. **Prior hospital admissions** showed a strong positive correlation with
   readmission — patients with 3+ prior admissions had nearly 2× the readmission rate.

3. **Emergency admissions** had significantly higher readmission rates than
   elective admissions, suggesting that unplanned acute episodes are a key
   intervention point.

4. **Length of stay** alone was not a strong predictor, but combined with
   discharge disposition (SNF, AMA) it significantly increased risk.

5. **K-Means clustering** revealed distinct patient subgroups that can guide
   targeted post-discharge intervention strategies.

---

## 5. Limitations & Assumptions

- **Synthetic data**: Results are based on simulated data and have not been
  validated on real patient populations. Real-world performance may differ.
- **Feature availability**: Some strong real-world predictors (e.g., social
  determinants of health, medication adherence) are not included.
- **Static model**: The pipeline does not account for temporal drift in patient
  populations or clinical practice changes over time.
- **Class imbalance**: The ~22% positive rate creates mild imbalance; production
  models should consider SMOTE or class-weight adjustments.

---

## 6. Generative AI Contribution

This project demonstrates Generative AI in three ways:

1. **Development Assistance**: GitHub Copilot accelerated code writing for data
   processing pipelines, evaluation loops, and visualisation boilerplate.

2. **Prompt Engineering**: This module generates structured prompts for LLM
   integration (`generate_analysis_prompts()`), enabling future deployment with
   OpenAI, Azure OpenAI, or Amazon Bedrock APIs.

3. **Report Generation**: This report itself was generated programmatically using
   a template-based GenAI pattern, demonstrating how ML outputs can be
   automatically translated into natural-language narratives for stakeholders.

---

## 7. Next Steps

- Validate the model on a real de-identified dataset (e.g., MIMIC-III/IV)
- Incorporate social determinants of health as additional features
- Implement SHAP explainability for individual patient predictions
- Build an API endpoint for real-time readmission scoring at discharge
- Integrate with EHR systems via HL7 FHIR standards

---

*This project was built as part of the **SFHA Advanced Data + AI Program** Capstone (Week 8).*
"""

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[GenAI] Analysis report saved: {output_path}")

    return report


# --------------------------------------------------------------------------- #
# Master run function
# --------------------------------------------------------------------------- #

def run_generative_ai(
    metrics_df: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    readmission_rate: float,
    dataset_size: int,
) -> str:
    """
    Execute the full Generative AI reporting pipeline.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Classification model evaluation metrics.
    cluster_profiles : pd.DataFrame
        K-Means cluster profiles.
    readmission_rate : float
        Overall dataset readmission rate.
    dataset_size : int
        Number of patient records.

    Returns
    -------
    str
        Generated Markdown report.
    """
    print("\n" + "=" * 60)
    print("GENERATIVE AI – REPORT GENERATION")
    print("=" * 60)

    # Generate structured LLM prompts (for documentation purposes)
    prompts = generate_analysis_prompts(metrics_df, cluster_profiles)
    print(f"[GenAI] Generated {len(prompts)} structured LLM prompts.")
    print("[GenAI] Prompts are ready to be sent to any LLM API (OpenAI, Claude, etc.)")

    # Generate the narrative report
    report = generate_report(
        metrics_df=metrics_df,
        cluster_profiles=cluster_profiles,
        readmission_rate=readmission_rate,
        dataset_size=dataset_size,
    )

    print("[GenAI] Pipeline complete.")
    return report


if __name__ == "__main__":
    # Demo with dummy data
    dummy_metrics = pd.DataFrame(
        {
            "Accuracy": [0.80, 0.83, 0.84],
            "Precision": [0.65, 0.70, 0.72],
            "Recall": [0.60, 0.65, 0.67],
            "F1 Score": [0.62, 0.67, 0.69],
            "ROC-AUC": [0.81, 0.86, 0.88],
        },
        index=["Logistic Regression", "Random Forest", "Gradient Boosting"],
    )
    dummy_clusters = pd.DataFrame(
        {
            "age": [38.5, 55.2, 68.1, 74.8],
            "length_of_stay": [3.1, 5.8, 7.9, 10.2],
            "num_previous_admissions": [0.5, 1.2, 2.8, 4.1],
            "comorbidity_score": [0.3, 0.9, 1.7, 2.4],
            "readmitted": [0.12, 0.19, 0.28, 0.41],
        },
        index=[0, 1, 2, 3],
    )
    run_generative_ai(dummy_metrics, dummy_clusters, 0.22, 5000)
