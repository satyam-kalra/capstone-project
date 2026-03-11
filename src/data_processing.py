"""
Data Processing Module
======================
Handles data cleansing, manipulation (feature engineering), and visualization
for the hospital patient readmission prediction project.

GenAI Usage Note:
    GitHub Copilot was used to suggest feature engineering strategies
    (e.g., age group binning thresholds, outlier capping logic) and to
    accelerate writing the boilerplate visualization code. All suggestions
    were reviewed and adapted for clinical relevance.
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "patient_readmission_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
CLEAN_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_data.csv")


# --------------------------------------------------------------------------- #
# 1. Data Loading
# --------------------------------------------------------------------------- #

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw patient dataset from a CSV file.

    Parameters
    ----------
    path : str
        File path to the raw CSV data.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame loaded from disk.
    """
    df = pd.read_csv(path)
    print(f"[Data Loading] Loaded {len(df)} rows × {len(df.columns)} columns from {path}")
    return df


# --------------------------------------------------------------------------- #
# 2. Data Cleansing
# --------------------------------------------------------------------------- #

def cleanse_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleansing steps:
      - Report and handle missing values
      - Remove duplicate rows
      - Fix data types
      - Cap numerical outliers using the IQR method

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleansed DataFrame.
    """
    df = df.copy()

    # --- Missing values ---
    missing = df.isnull().sum()
    if missing.any():
        print("[Cleansing] Missing values detected:")
        print(missing[missing > 0])
        # Fill numeric columns with median, categorical with mode
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        print("[Cleansing] No missing values found.")

    # --- Duplicates ---
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates()
        print(f"[Cleansing] Removed {n_dupes} duplicate rows.")
    else:
        print("[Cleansing] No duplicate rows found.")

    # --- Data types ---
    binary_cols = ["has_diabetes", "has_hypertension", "has_heart_disease", "readmitted"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # --- Outlier capping (IQR method on numerical columns) ---
    numerical_cols = [
        "age", "num_procedures", "num_medications",
        "num_lab_tests", "length_of_stay", "num_previous_admissions",
    ]
    for col in numerical_cols:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_outliers > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                print(f"[Cleansing] Capped {n_outliers} outliers in '{col}'.")

    print(f"[Cleansing] Final dataset shape: {df.shape}")
    return df


# --------------------------------------------------------------------------- #
# 3. Feature Engineering (Data Manipulation)
# --------------------------------------------------------------------------- #

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering:
      - Create age group bins
      - Compute comorbidity score
      - Encode categorical variables
      - Scale numerical features

    Parameters
    ----------
    df : pd.DataFrame
        Cleansed DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features ready for modelling.
    """
    df = df.copy()

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 75, 100],
        labels=["18-30", "31-45", "46-60", "61-75", "76+"],
    )

    # Comorbidity score (sum of binary flags)
    df["comorbidity_score"] = df["has_diabetes"] + df["has_hypertension"] + df["has_heart_disease"]

    # Encode categorical variables (store each encoder for potential future use)
    encoders = {}
    cat_cols = ["gender", "admission_type", "diagnosis", "discharge_disposition"]
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Age group encoded
    le_age = LabelEncoder()
    df["age_group_encoded"] = le_age.fit_transform(df["age_group"].astype(str))
    encoders["age_group"] = le_age

    # Normalise numerical features (stored with '_scaled' suffix)
    scaler = MinMaxScaler()
    scale_cols = [
        "age", "num_procedures", "num_medications",
        "num_lab_tests", "length_of_stay", "num_previous_admissions",
        "comorbidity_score",
    ]
    scaled_values = scaler.fit_transform(df[scale_cols])
    for i, col in enumerate(scale_cols):
        df[f"{col}_scaled"] = scaled_values[:, i]

    print(f"[Feature Engineering] Added features. Total columns: {len(df.columns)}")
    return df


# --------------------------------------------------------------------------- #
# 4. Visualizations
# --------------------------------------------------------------------------- #

def _ensure_plot_dir() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_readmission_distribution(df: pd.DataFrame) -> None:
    """Plot the distribution of the readmission target variable."""
    _ensure_plot_dir()
    counts = df["readmitted"].value_counts()
    labels = ["Not Readmitted", "Readmitted"]
    colors = ["#4CAF50", "#F44336"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Readmission Distribution", fontsize=15, fontweight="bold")

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Patient Count by Readmission Status")
    axes[0].set_ylabel("Number of Patients")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[1].set_title("Readmission Rate")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_readmission_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot a correlation heatmap of numerical features."""
    _ensure_plot_dir()
    num_cols = [
        "age", "num_procedures", "num_medications", "num_lab_tests",
        "length_of_stay", "num_previous_admissions",
        "has_diabetes", "has_hypertension", "has_heart_disease",
        "comorbidity_score", "readmitted",
    ]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr()

    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {path}")


def plot_age_group_distribution(df: pd.DataFrame) -> None:
    """Plot readmission rates across age groups."""
    _ensure_plot_dir()
    if "age_group" not in df.columns:
        df = df.copy()
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 75, 100],
            labels=["18-30", "31-45", "46-60", "61-75", "76+"],
        )

    group_stats = (
        df.groupby("age_group", observed=True)["readmitted"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    group_stats.columns = ["age_group", "total", "readmitted", "rate"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Age Group Analysis", fontsize=15, fontweight="bold")

    palette = sns.color_palette("Blues_d", len(group_stats))
    axes[0].bar(group_stats["age_group"].astype(str), group_stats["total"],
                color=palette, edgecolor="white")
    axes[0].set_title("Patient Count by Age Group")
    axes[0].set_xlabel("Age Group")
    axes[0].set_ylabel("Count")

    axes[1].bar(group_stats["age_group"].astype(str), group_stats["rate"] * 100,
                color=sns.color_palette("Reds_d", len(group_stats)), edgecolor="white")
    axes[1].set_title("Readmission Rate by Age Group (%)")
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("Readmission Rate (%)")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_age_group_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {path}")


def plot_length_of_stay_analysis(df: pd.DataFrame) -> None:
    """Plot length of stay distributions by readmission status."""
    _ensure_plot_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Length of Stay Analysis", fontsize=15, fontweight="bold")

    # Histogram with KDE by readmission status
    for val, label, color in [(0, "Not Readmitted", "#4CAF50"), (1, "Readmitted", "#F44336")]:
        subset = df[df["readmitted"] == val]["length_of_stay"]
        axes[0].hist(subset, bins=20, alpha=0.6, label=label, color=color, edgecolor="white")
    axes[0].set_title("Length of Stay by Readmission Status")
    axes[0].set_xlabel("Length of Stay (days)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Box plot by diagnosis
    top_diagnoses = df["diagnosis"].value_counts().head(5).index
    subset = df[df["diagnosis"].isin(top_diagnoses)]
    subset.boxplot(
        column="length_of_stay",
        by="diagnosis",
        ax=axes[1],
        patch_artist=True,
    )
    axes[1].set_title("Length of Stay by Top Diagnoses")
    axes[1].set_xlabel("Diagnosis")
    axes[1].set_ylabel("Days")
    plt.sca(axes[1])
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "04_length_of_stay_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {path}")


def plot_readmission_by_diagnosis(df: pd.DataFrame) -> None:
    """Plot readmission rates broken down by primary diagnosis."""
    _ensure_plot_dir()
    diag_stats = (
        df.groupby("diagnosis")["readmitted"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )
    diag_stats.columns = ["diagnosis", "count", "rate"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        diag_stats["diagnosis"],
        diag_stats["rate"] * 100,
        color=sns.color_palette("RdYlGn_r", len(diag_stats)),
        edgecolor="white",
    )
    ax.set_title("30-Day Readmission Rate by Diagnosis", fontsize=14, fontweight="bold")
    ax.set_xlabel("Readmission Rate (%)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for bar, count in zip(bars, diag_stats["count"]):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"n={count}",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "05_readmission_by_diagnosis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {path}")


def generate_all_visualizations(df: pd.DataFrame) -> None:
    """Run all five visualization functions."""
    print("\n[Visualizations] Generating plots...")
    plot_readmission_distribution(df)
    plot_correlation_heatmap(df)
    plot_age_group_distribution(df)
    plot_length_of_stay_analysis(df)
    plot_readmission_by_diagnosis(df)
    print("[Visualizations] All plots saved to", PLOTS_DIR)


# --------------------------------------------------------------------------- #
# 5. Save cleaned data
# --------------------------------------------------------------------------- #

def save_cleaned_data(df: pd.DataFrame, path: str = CLEAN_DATA_PATH) -> None:
    """
    Save the processed (cleansed + feature-engineered) DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame.
    path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Save] Cleaned data saved to {path}")


# --------------------------------------------------------------------------- #
# 6. Master run function
# --------------------------------------------------------------------------- #

def run_data_processing(raw_data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Execute the full data processing pipeline.

    Steps:
      1. Load raw data
      2. Cleanse data
      3. Engineer features
      4. Generate visualizations
      5. Save cleaned data

    Parameters
    ----------
    raw_data_path : str
        Path to the raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        Fully processed DataFrame ready for modelling.
    """
    df = load_data(raw_data_path)
    df = cleanse_data(df)
    df = engineer_features(df)
    generate_all_visualizations(df)
    save_cleaned_data(df)
    print("\n[Data Processing] Pipeline complete.")
    return df


if __name__ == "__main__":
    run_data_processing()
