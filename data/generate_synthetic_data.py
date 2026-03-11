"""
Synthetic Healthcare Data Generator
=====================================
Generates a realistic synthetic dataset of ~5000 hospital patients for the
30-day readmission prediction capstone project.

GenAI Usage Note:
    GitHub Copilot was used to help design the probability distributions for
    readmission risk factors, making the synthetic data more clinically
    realistic (e.g., older patients with more comorbidities have higher
    readmission rates).
"""

import numpy as np
import pandas as pd
import os


def generate_patient_data(n_patients: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic healthcare dataset for hospital readmission prediction.

    Parameters
    ----------
    n_patients : int
        Number of synthetic patient records to generate (default: 5000).
    random_seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    pd.DataFrame
        DataFrame containing synthetic patient records with realistic
        distributions and correlations between risk factors and readmission.
    """
    rng = np.random.default_rng(random_seed)

    # --- Patient demographics ---
    patient_ids = [f"P{str(i).zfill(5)}" for i in range(1, n_patients + 1)]
    ages = rng.integers(18, 91, size=n_patients)
    genders = rng.choice(["Male", "Female"], size=n_patients, p=[0.48, 0.52])

    # --- Admission details ---
    admission_types = rng.choice(
        ["Emergency", "Urgent", "Elective"],
        size=n_patients,
        p=[0.45, 0.30, 0.25],
    )
    diagnoses = rng.choice(
        [
            "Heart Disease",
            "Diabetes",
            "Pneumonia",
            "COPD",
            "Kidney Disease",
            "Stroke",
            "Hip Fracture",
            "Sepsis",
        ],
        size=n_patients,
        p=[0.22, 0.20, 0.14, 0.12, 0.12, 0.08, 0.06, 0.06],
    )

    # --- Clinical metrics (partially correlated with disease severity) ---
    # Comorbidities (correlated with age) — define before using for clinical metrics
    age_factor = (ages - 18) / 72  # normalised 0-1
    has_diabetes = (rng.random(n_patients) < 0.15 + 0.25 * age_factor).astype(int)
    has_hypertension = (rng.random(n_patients) < 0.20 + 0.35 * age_factor).astype(int)
    has_heart_disease = (rng.random(n_patients) < 0.10 + 0.30 * age_factor).astype(int)
    comorbidity_total = has_diabetes + has_hypertension + has_heart_disease

    # Clinical severity proxy: higher for complex patients
    severity_base = 0.3 * age_factor + 0.2 * (comorbidity_total / 3.0)

    # Diagnoses with high severity get more procedures/meds
    high_severity_diag = np.isin(diagnoses, ["Heart Disease", "Stroke", "Sepsis", "Kidney Disease"])

    num_procedures = np.clip(
        rng.integers(0, 6, n_patients) + (high_severity_diag * rng.integers(0, 5, n_patients)),
        0, 12,
    ).astype(int)
    num_medications = np.clip(
        rng.integers(1, 10, n_patients)
        + (comorbidity_total * rng.integers(1, 4, n_patients)),
        1, 25,
    ).astype(int)
    num_lab_tests = np.clip(
        rng.integers(1, 12, n_patients)
        + (high_severity_diag * rng.integers(0, 8, n_patients)),
        1, 30,
    ).astype(int)
    # Length of stay correlated with severity
    length_of_stay = np.clip(
        rng.integers(1, 8, n_patients)
        + np.round(severity_base * 10 + high_severity_diag * rng.integers(0, 6, n_patients)).astype(int),
        1, 25,
    ).astype(int)
    # Prior admissions correlated with age and comorbidities
    num_previous_admissions = np.clip(
        rng.integers(0, 4, n_patients)
        + np.round(age_factor * 3 + comorbidity_total).astype(int)
        + rng.integers(0, 2, n_patients),
        0, 10,
    ).astype(int)

    # --- Discharge disposition ---
    discharge_disposition = rng.choice(
        ["Home", "SNF", "Rehab", "Home Health", "AMA"],
        size=n_patients,
        p=[0.55, 0.18, 0.14, 0.10, 0.03],
    )

    # --- Readmission target (realistic ~20-25% rate, target AUC ~0.72) ---
    # Use a logistic model to generate clear, clinically plausible readmission probabilities.
    # Coefficients chosen to yield ~22% average readmission rate and meaningful AUC.

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # Diagnosis risk scores (log-odds contribution)
    diag_logodds = {
        "Heart Disease": 0.60, "Stroke": 0.55, "Kidney Disease": 0.45,
        "COPD": 0.40, "Sepsis": 0.35, "Diabetes": 0.20,
        "Pneumonia": 0.15, "Hip Fracture": 0.10,
    }

    log_odds = (
        -5.20                                                            # intercept (tuned for ~22% readmission rate)
        + 0.025 * ages                                                   # age
        + 0.50 * (admission_types == "Emergency").astype(float)          # emergency
        + 0.30 * (admission_types == "Urgent").astype(float)             # urgent
        + 0.80 * has_heart_disease                                       # comorbidity
        + 0.55 * has_diabetes                                            # comorbidity
        + 0.40 * has_hypertension                                        # comorbidity
        + 0.28 * num_previous_admissions                                 # utilisation
        + 0.75 * (discharge_disposition == "AMA").astype(float)          # AMA discharge
        + 0.25 * (discharge_disposition == "SNF").astype(float)          # SNF
        + np.array([diag_logodds.get(d, 0.15) for d in diagnoses])       # diagnosis
    )
    risk_prob = _sigmoid(log_odds)
    readmitted = (rng.random(n_patients) < risk_prob).astype(int)

    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": ages,
            "gender": genders,
            "admission_type": admission_types,
            "diagnosis": diagnoses,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "num_lab_tests": num_lab_tests,
            "length_of_stay": length_of_stay,
            "num_previous_admissions": num_previous_admissions,
            "has_diabetes": has_diabetes,
            "has_hypertension": has_hypertension,
            "has_heart_disease": has_heart_disease,
            "discharge_disposition": discharge_disposition,
            "readmitted": readmitted,
        }
    )

    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the generated dataset to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The synthetic patient dataset.
    output_path : str
        File path where the CSV will be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"Readmission rate: {df['readmitted'].mean():.1%}")


if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "patient_readmission_data.csv")
    dataset = generate_patient_data(n_patients=5000)
    save_dataset(dataset, output_file)
    print("\nSample records:")
    print(dataset.head())
