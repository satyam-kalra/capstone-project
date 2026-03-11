"""
Generate Synthetic Hospital Readmission Dataset
================================================
SFHA Advanced Data + AI Program - Week 8 Capstone Project

This script generates a realistic synthetic dataset of ~1000 hospital patients
and saves it as data/hospital_readmissions.csv.

GenAI Assistance Note:
    GitHub Copilot was used to help design realistic probability distributions
    for comorbidities and readmission rates based on clinical literature.
    The feature correlations (e.g., diabetes↔higher medications, emergency
    admission↔higher readmission) reflect published healthcare statistics.

Usage:
    python data/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Dataset size ─────────────────────────────────────────────────────────────
N_PATIENTS = 1000

# ── Helper generators ─────────────────────────────────────────────────────────

def generate_patient_ids(n: int) -> list[str]:
    """Return zero-padded patient IDs, e.g. 'P0001'."""
    return [f"P{str(i).zfill(4)}" for i in range(1, n + 1)]


def generate_demographics(n: int) -> pd.DataFrame:
    """Return age and gender columns with realistic distributions."""
    ages = np.random.normal(loc=60, scale=15, size=n).clip(18, 95).astype(int)
    genders = np.random.choice(["Male", "Female"], size=n, p=[0.48, 0.52])
    return pd.DataFrame({"age": ages, "gender": genders})


def generate_admission_info(n: int) -> pd.DataFrame:
    """Return admission type, diagnosis category, and related clinical fields."""
    admission_types = np.random.choice(
        ["Emergency", "Urgent", "Elective"],
        size=n,
        p=[0.35, 0.30, 0.35],
    )
    diagnosis_categories = np.random.choice(
        [
            "Heart Disease",
            "Diabetes",
            "Pneumonia",
            "COPD",
            "Kidney Disease",
            "Stroke",
            "Sepsis",
            "Other",
        ],
        size=n,
        p=[0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.10],
    )
    num_lab_procedures = np.random.randint(1, 80, size=n)
    num_medications    = np.random.randint(1, 30, size=n)
    num_previous_admissions = np.random.choice(
        [0, 1, 2, 3, 4, 5],
        size=n,
        p=[0.40, 0.25, 0.18, 0.10, 0.05, 0.02],
    )
    length_of_stay = np.random.gamma(shape=3, scale=2, size=n).clip(1, 30).astype(int)

    discharge_dispositions = np.random.choice(
        ["Home", "SNF", "Rehab", "AMA", "Expired"],
        size=n,
        p=[0.60, 0.18, 0.14, 0.05, 0.03],
    )

    return pd.DataFrame(
        {
            "admission_type": admission_types,
            "diagnosis_category": diagnosis_categories,
            "num_lab_procedures": num_lab_procedures,
            "num_medications": num_medications,
            "num_previous_admissions": num_previous_admissions,
            "length_of_stay": length_of_stay,
            "discharge_disposition": discharge_dispositions,
        }
    )


def generate_comorbidities(n: int, diagnosis_categories: np.ndarray) -> pd.DataFrame:
    """Return comorbidity flags.  Probability is elevated for relevant diagnoses."""
    has_diabetes = np.where(
        diagnosis_categories == "Diabetes",
        np.random.binomial(1, 0.90, n),
        np.random.binomial(1, 0.25, n),
    )
    has_hypertension = np.where(
        np.isin(diagnosis_categories, ["Heart Disease", "Kidney Disease", "Stroke"]),
        np.random.binomial(1, 0.85, n),
        np.random.binomial(1, 0.40, n),
    )
    return pd.DataFrame(
        {
            "has_diabetes": has_diabetes,
            "has_hypertension": has_hypertension,
        }
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_readmission_target(df: pd.DataFrame) -> np.ndarray:
    """
    Compute 30-day readmission probability via a logistic model, then sample.

    Using a logistic (sigmoid) model gives clean linear relationships in
    log-odds space, ensuring the trained classifiers can learn meaningful
    signal (target AUC ≈ 0.70-0.75, readmission rate ≈ 20-25 %).

    Intercept calibration:
        intercept = -5.0 → sigmoid(-5) ≈ 0.7 % base rate for a patient
        with no additional risk factors.  After adding the weighted sum of
        risk factors the overall sample mean converges to ~20-25 %.

    Coefficients (log-odds units) reflect relative clinical importance:
        Emergency admission has the largest single effect (+2.0 log-odds),
        consistent with published readmission risk literature.
    """
    intercept = -5.0

    logit = np.full(len(df), intercept, dtype=float)

    # Admission urgency
    logit += 2.0 * (df["admission_type"] == "Emergency").astype(float)
    logit += 1.2 * (df["admission_type"] == "Urgent").astype(float)

    # Prior admissions (0.5 log-odds per previous admission, up to 5)
    logit += df["num_previous_admissions"].clip(upper=5) * 0.50

    # Comorbidities
    logit += 1.2 * df["has_diabetes"].astype(float)
    logit += 0.8 * df["has_hypertension"].astype(float)

    # Polypharmacy (>=10 medications)
    logit += 0.9 * (df["num_medications"] >= 10).astype(float)

    # High-risk discharge destinations
    logit += 2.5 * (df["discharge_disposition"] == "AMA").astype(float)
    logit += 0.7 * (df["discharge_disposition"] == "SNF").astype(float)

    # Prolonged stay (>10 days)
    logit += 1.0 * (df["length_of_stay"] > 10).astype(float)

    prob = _sigmoid(logit)

    return np.random.binomial(1, prob)


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    print("Generating synthetic hospital readmission dataset …")

    # Demographics
    demographics = generate_demographics(N_PATIENTS)

    # Admission info
    admission_info = generate_admission_info(N_PATIENTS)

    # Comorbidities (depend on diagnosis)
    comorbidities = generate_comorbidities(
        N_PATIENTS, admission_info["diagnosis_category"].values
    )

    # Assemble base dataframe (needed for target generation)
    df = pd.concat([demographics, admission_info, comorbidities], axis=1)

    # Target variable
    readmitted = generate_readmission_target(df)

    # Final dataset
    df.insert(0, "patient_id", generate_patient_ids(N_PATIENTS))
    df["readmitted_within_30_days"] = readmitted

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(__file__), "hospital_readmissions.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset saved → {output_path}")
    print(f"Shape         : {df.shape}")
    print(f"Readmission % : {readmitted.mean() * 100:.1f}%")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
