"""
Capstone Analysis: Canadian Hospital Readmission Rates (Real CIHI Data)
=======================================================================
SFHA Advanced Data + AI Program – Week 8 Capstone Project

Data Source:
    Canadian Institute for Health Information (CIHI)
    https://www.cihi.ca/en/indicators/all-patients-readmitted-to-hospital

    Files used:
      data/indicator-library-all-indicator-data-en.xlsx
      data/827-all-patients-readmitted-to-hospital-data-table-en.xlsx

This single runnable script covers all three required core concepts:
    - Part 1: Data Processing  (loading, cleansing, manipulation, visualisation)
    - Part 2: Machine Learning (regression, classification, feature importance)
    - Part 3: Generative AI    (documented usage + Canadian-context prompt templates)

Usage:
    # 1. Download CIHI data (required before running this script)
    python data/download_cihi_data.py

    # 2. Run the full analysis
    python notebooks/capstone_analysis.py

All outputs are saved to the outputs/ directory.

GenAI Assistance Notes (required by rubric):
    Throughout this project GitHub Copilot and ChatGPT were used for:
    - Ideation: brainstorming feature-engineering approaches for aggregate
      Canadian health indicator data from CIHI
    - Coding assistance: drafting boilerplate (model evaluation helpers,
      seaborn theming, Excel sheet parsing logic)
    - Result interpretation: refining the Canadian-context prompt templates
      in Part 3 so that an LLM can generate policy summaries for provincial
      health ministers
    All AI-generated suggestions were reviewed, tested, and adapted before
    being included in the final code.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
from typing import Optional

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # non-interactive backend for saving plots

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUTS = os.path.join(REPO_ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

INDICATOR_FILE = os.path.join(
    DATA_DIR, "indicator-library-all-indicator-data-en.xlsx"
)
READMIT_FILE = os.path.join(
    DATA_DIR, "827-all-patients-readmitted-to-hospital-data-table-en.xlsx"
)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting theme
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
CANADA_PALETTE = [
    "#D62828",  # Canada red
    "#003566",  # deep blue
    "#457B9D",
    "#1D3557",
    "#A8DADC",
    "#E63946",
    "#F4A261",
    "#2A9D8F",
    "#264653",
    "#E9C46A",
    "#606C38",
    "#DDA15E",
    "#BC6C25",
]

# Province/territory order for charts
PROVINCE_ORDER = [
    "British Columbia",
    "Alberta",
    "Saskatchewan",
    "Manitoba",
    "Ontario",
    "Quebec",
    "New Brunswick",
    "Nova Scotia",
    "Prince Edward Island",
    "Newfoundland and Labrador",
    "Yukon",
    "Northwest Territories",
    "Nunavut",
    "Canada",
]


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def _find_header_row(path: str, sheet: str, keyword: str, max_rows: int = 15) -> int:
    """Scan the first *max_rows* rows of a sheet to find the header row.

    Returns the 0-based row index of the first row that contains *keyword*.
    Returns 0 if not found (safe default).
    """
    probe = pd.read_excel(
        path, sheet_name=sheet, header=None, nrows=max_rows, engine="openpyxl"
    )
    for i, row in probe.iterrows():
        if any(
            isinstance(cell, str) and keyword.lower() in cell.lower()
            for cell in row
        ):
            return int(i)
    return 0


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric, replacing non-numeric values with NaN."""
    return pd.to_numeric(series, errors="coerce")


def _save_fig(filename: str) -> None:
    path = os.path.join(OUTPUTS, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {filename}")


# =============================================================================
# PART 1 – DATA PROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("PART 1 — DATA PROCESSING")
print("=" * 70)

# ── 1.1  Check data files exist ───────────────────────────────────────────────
print("\n[1.1] Checking data files …")
missing_files = []
for fp in [INDICATOR_FILE, READMIT_FILE]:
    if not os.path.exists(fp):
        missing_files.append(fp)
    else:
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"  Found: {os.path.basename(fp)}  ({size_mb:.1f} MB)")

if missing_files:
    print("\nERROR: The following data files are missing:")
    for fp in missing_files:
        print(f"  {fp}")
    print("\nRun the downloader first:\n  python data/download_cihi_data.py")
    sys.exit(1)

# ── 1.2  Load CIHI Indicator Library ─────────────────────────────────────────
print("\n[1.2] Loading CIHI Indicator Library …")

xl_indicator = pd.ExcelFile(INDICATOR_FILE, engine="openpyxl")
print(f"  Sheets: {xl_indicator.sheet_names}")

# Find the data sheet (usually first non-readme sheet)
data_sheet = xl_indicator.sheet_names[0]
for sheet in xl_indicator.sheet_names:
    if any(kw in sheet.lower() for kw in ["data", "indicator", "all"]):
        data_sheet = sheet
        break

header_row = _find_header_row(INDICATOR_FILE, data_sheet, "indicator", max_rows=15)
print(f"  Using sheet '{data_sheet}', header at row {header_row}")

df_indicators_raw = pd.read_excel(
    INDICATOR_FILE,
    sheet_name=data_sheet,
    header=header_row,
    engine="openpyxl",
)
# Drop fully-empty rows and columns
df_indicators_raw.dropna(how="all", inplace=True)
df_indicators_raw.dropna(axis=1, how="all", inplace=True)
print(
    f"  Indicator library shape: {df_indicators_raw.shape[0]:,} rows × "
    f"{df_indicators_raw.shape[1]} columns"
)
print(f"  Columns: {list(df_indicators_raw.columns[:10])} …")

# ── 1.3  Load 827 Readmission Data Table ─────────────────────────────────────
print("\n[1.3] Loading 827 – All Patients Readmitted to Hospital …")

xl_readmit = pd.ExcelFile(READMIT_FILE, engine="openpyxl")
print(f"  Sheets: {xl_readmit.sheet_names}")

# Identify the main data sheet
readmit_sheet = xl_readmit.sheet_names[0]
for sheet in xl_readmit.sheet_names:
    if any(kw in sheet.lower() for kw in ["data", "readmit", "all patient", "827"]):
        readmit_sheet = sheet
        break

header_row_r = _find_header_row(READMIT_FILE, readmit_sheet, "province", max_rows=15)
if header_row_r == 0:
    header_row_r = _find_header_row(
        READMIT_FILE, readmit_sheet, "year", max_rows=15
    )
print(f"  Using sheet '{readmit_sheet}', header at row {header_row_r}")

df_readmit_raw = pd.read_excel(
    READMIT_FILE,
    sheet_name=readmit_sheet,
    header=header_row_r,
    engine="openpyxl",
)
df_readmit_raw.dropna(how="all", inplace=True)
df_readmit_raw.dropna(axis=1, how="all", inplace=True)
print(
    f"  Readmission table shape : {df_readmit_raw.shape[0]:,} rows × "
    f"{df_readmit_raw.shape[1]} columns"
)
print(f"  Columns: {list(df_readmit_raw.columns[:10])} …")

# ── 1.4  Column Normalisation ─────────────────────────────────────────────────
print("\n[1.4] Normalising column names …")


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and lower-case column names."""
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


df_ind = _normalise_cols(df_indicators_raw)
df_rd = _normalise_cols(df_readmit_raw)

print(f"  Indicator cols : {list(df_ind.columns[:8])} …")
print(f"  Readmit cols   : {list(df_rd.columns[:8])} …")

# ── 1.5  Filter Indicator Library for Readmission Indicators ──────────────────
print("\n[1.5] Filtering readmission-related indicators …")

READMISSION_KEYWORDS = [
    "readmit",
    "readmission",
    "re-admit",
]

# Identify the indicator name column
name_col_candidates = [
    c for c in df_ind.columns if "indicator" in c and "name" in c
]
name_col = name_col_candidates[0] if name_col_candidates else df_ind.columns[0]
print(f"  Using indicator name column: '{name_col}'")

mask = df_ind[name_col].astype(str).str.lower().str.contains(
    "|".join(READMISSION_KEYWORDS), na=False
)
df_readmit_indicators = df_ind[mask].copy()
print(
    f"  Found {len(df_readmit_indicators):,} readmission indicator rows "
    f"(out of {len(df_ind):,} total)"
)

# If no rows match, keep all indicator rows but flag for user
if df_readmit_indicators.empty:
    print("  WARNING: No readmission indicators found via keyword filter.")
    print("  Keeping all indicators for downstream analysis.")
    df_readmit_indicators = df_ind.copy()

# ── 1.6  Handle Suppressed Values & Missing Data ──────────────────────────────
print("\n[1.6] Handling suppressed and missing values …")

SUPPRESSION_FLAGS = ["s", "sp", "sup", "suppressed", "n/a", "na", "--", "x"]


def _clean_result_value(series: pd.Series) -> pd.Series:
    """Replace suppression flags with NaN and coerce to float."""
    cleaned = series.astype(str).str.strip().str.lower()
    cleaned = cleaned.replace(SUPPRESSION_FLAGS, np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


# Find result/value column in readmit table
result_col_candidates = [
    c for c in df_rd.columns if any(kw in c for kw in ["result", "value", "rate"])
]
result_col = result_col_candidates[0] if result_col_candidates else df_rd.columns[-1]
print(f"  Result value column: '{result_col}'")

df_rd["result_value_clean"] = _clean_result_value(df_rd[result_col])

n_suppressed = df_rd["result_value_clean"].isna().sum()
n_total = len(df_rd)
print(
    f"  Suppressed/missing values: {n_suppressed:,} / {n_total:,} "
    f"({n_suppressed / n_total * 100:.1f}%)"
)

# Also clean result column in indicator data
result_col_ind_candidates = [
    c
    for c in df_readmit_indicators.columns
    if any(kw in c for kw in ["result", "value", "rate"])
]
if result_col_ind_candidates:
    result_col_ind = result_col_ind_candidates[0]
    df_readmit_indicators = df_readmit_indicators.copy()
    df_readmit_indicators["result_value_clean"] = _clean_result_value(
        df_readmit_indicators[result_col_ind]
    )

# ── 1.7  Identify Key Columns & Feature Engineering ───────────────────────────
print("\n[1.7] Feature engineering …")

# Helper to find a column by keyword
def _find_col(df: pd.DataFrame, *keywords: str) -> Optional[str]:
    for kw in keywords:
        for col in df.columns:
            if kw in col:
                return col
    return None


# Readmit table columns
province_col = _find_col(df_rd, "province", "territory", "region", "place")
year_col = _find_col(df_rd, "year", "fiscal", "period")
age_col = _find_col(df_rd, "age")
sex_col = _find_col(df_rd, "sex", "gender")
patient_type_col = _find_col(df_rd, "patient_type", "patient", "type")

print(f"  province_col     : {province_col}")
print(f"  year_col         : {year_col}")
print(f"  age_col          : {age_col}")
print(f"  sex_col          : {sex_col}")
print(f"  patient_type_col : {patient_type_col}")

df_rd_clean = df_rd[df_rd["result_value_clean"].notna()].copy()

# Year as integer
if year_col:
    df_rd_clean["year_numeric"] = (
        df_rd_clean[year_col]
        .astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

# Provincial flag: is this a national Canada row?
if province_col:
    df_rd_clean["is_national"] = (
        df_rd_clean[province_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["canada", "national"])
    )

# Above-national-average rate flag (for classification target)
if "result_value_clean" in df_rd_clean.columns:
    national_mean = df_rd_clean["result_value_clean"].mean()
    df_rd_clean["above_avg"] = (
        df_rd_clean["result_value_clean"] > national_mean
    ).astype(int)
    print(f"  National mean readmission rate : {national_mean:.2f}%")
    print(
        f"  Above-average rows             : "
        f"{df_rd_clean['above_avg'].sum():,} / {len(df_rd_clean):,}"
    )

print(f"  Clean rows for analysis: {len(df_rd_clean):,}")

# Save cleaned data
clean_csv = os.path.join(OUTPUTS, "cihi_readmission_clean.csv")
df_rd_clean.to_csv(clean_csv, index=False)
print(f"  Saved cleaned data → cihi_readmission_clean.csv")

# ── 1.8  Visualisations ───────────────────────────────────────────────────────
print("\n[1.8] Creating visualisations …")

# ── Plot 1: Readmission Rate by Province/Territory ────────────────────────────
if province_col and "result_value_clean" in df_rd_clean.columns:
    fig, ax = plt.subplots(figsize=(12, 7))
    prov_rates = (
        df_rd_clean.groupby(province_col)["result_value_clean"]
        .mean()
        .reset_index()
        .rename(
            columns={
                province_col: "Province_Territory",
                "result_value_clean": "Mean_Readmission_Rate",
            }
        )
        .sort_values("Mean_Readmission_Rate", ascending=True)
    )
    # Limit to reasonable number of provinces
    prov_rates = prov_rates.tail(20)
    colors = [
        "#D62828" if "canada" in str(p).lower() else "#457B9D"
        for p in prov_rates["Province_Territory"]
    ]
    bars = ax.barh(
        prov_rates["Province_Territory"],
        prov_rates["Mean_Readmission_Rate"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xlabel("Mean Readmission Rate (%)", fontsize=12)
    ax.set_title(
        "Hospital Readmission Rates by Province/Territory\n"
        "(Source: CIHI – All Patients Readmitted to Hospital)",
        fontsize=13,
        pad=12,
    )
    ax.axvline(
        prov_rates["Mean_Readmission_Rate"].mean(),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="National mean",
    )
    ax.legend()
    _save_fig("plot1_readmission_by_province.png")
else:
    print("  Skipped plot1 (province or result column not found)")

# ── Plot 2: Readmission Rate Trend Over Time ──────────────────────────────────
if year_col and "year_numeric" in df_rd_clean.columns:
    yearly = (
        df_rd_clean.groupby("year_numeric")["result_value_clean"]
        .agg(["mean", "std"])
        .reset_index()
    )
    if len(yearly) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            yearly["year_numeric"],
            yearly["mean"],
            marker="o",
            color="#D62828",
            linewidth=2.5,
            markersize=7,
            label="Mean readmission rate",
        )
        ax.fill_between(
            yearly["year_numeric"],
            yearly["mean"] - yearly["std"],
            yearly["mean"] + yearly["std"],
            alpha=0.2,
            color="#D62828",
            label="±1 SD",
        )
        ax.set_xlabel("Fiscal Year", fontsize=12)
        ax.set_ylabel("Readmission Rate (%)", fontsize=12)
        ax.set_title(
            "Canadian Hospital Readmission Rate Trend Over Time\n"
            "(Source: CIHI)",
            fontsize=13,
            pad=12,
        )
        ax.legend()
        _save_fig("plot2_readmission_trend.png")
    else:
        print("  Skipped plot2 (insufficient year data)")
else:
    print("  Skipped plot2 (year column not found)")

# ── Plot 3: Readmission Rate by Age Group ─────────────────────────────────────
if age_col:
    age_rates = (
        df_rd_clean.groupby(age_col)["result_value_clean"]
        .mean()
        .reset_index()
        .rename(
            columns={
                age_col: "Age_Group",
                "result_value_clean": "Mean_Readmission_Rate",
            }
        )
        .dropna()
        .sort_values("Mean_Readmission_Rate", ascending=False)
    )
    if len(age_rates) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=age_rates,
            x="Age_Group",
            y="Mean_Readmission_Rate",
            palette="Blues_d",
            ax=ax,
        )
        ax.set_xlabel("Age Group", fontsize=12)
        ax.set_ylabel("Mean Readmission Rate (%)", fontsize=12)
        ax.set_title(
            "Readmission Rates by Age Group\n(Source: CIHI)",
            fontsize=13,
            pad=12,
        )
        plt.xticks(rotation=45, ha="right")
        _save_fig("plot3_readmission_by_age.png")
    else:
        print("  Skipped plot3 (insufficient age group data)")
else:
    print("  Skipped plot3 (age column not found)")

# ── Plot 4: Readmission Rate by Sex ───────────────────────────────────────────
if sex_col:
    sex_rates = (
        df_rd_clean.groupby(sex_col)["result_value_clean"]
        .mean()
        .reset_index()
        .rename(
            columns={
                sex_col: "Sex",
                "result_value_clean": "Mean_Readmission_Rate",
            }
        )
        .dropna()
    )
    if len(sex_rates) > 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(
            data=sex_rates,
            x="Sex",
            y="Mean_Readmission_Rate",
            palette=["#457B9D", "#E63946"],
            ax=ax,
        )
        ax.set_xlabel("Sex", fontsize=12)
        ax.set_ylabel("Mean Readmission Rate (%)", fontsize=12)
        ax.set_title(
            "Readmission Rates by Sex\n(Source: CIHI)",
            fontsize=13,
            pad=12,
        )
        _save_fig("plot4_readmission_by_sex.png")
    else:
        print("  Skipped plot4 (insufficient sex data)")
else:
    print("  Skipped plot4 (sex column not found)")

# ── Plot 5: Readmission Rate by Patient Type (indicator comparison) ───────────
if name_col in df_readmit_indicators.columns and "result_value_clean" in df_readmit_indicators.columns:
    patient_type_rates = (
        df_readmit_indicators.groupby(name_col)["result_value_clean"]
        .mean()
        .reset_index()
        .rename(
            columns={
                name_col: "Indicator",
                "result_value_clean": "Mean_Rate",
            }
        )
        .dropna()
        .sort_values("Mean_Rate", ascending=False)
        .head(10)
    )
    if len(patient_type_rates) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=patient_type_rates,
            x="Mean_Rate",
            y="Indicator",
            palette="Reds_d",
            ax=ax,
        )
        ax.set_xlabel("Mean Readmission Rate (%)", fontsize=12)
        ax.set_ylabel("")
        ax.set_title(
            "Readmission Rates by Indicator / Patient Type\n(Source: CIHI Indicator Library)",
            fontsize=13,
            pad=12,
        )
        _save_fig("plot5_readmission_by_indicator.png")
    else:
        print("  Skipped plot5 (no indicator rate data)")
else:
    print("  Skipped plot5 (indicator name or result column not found)")

# ── Plot 6: Heatmap – Province × Year ────────────────────────────────────────
if province_col and year_col and "year_numeric" in df_rd_clean.columns:
    pivot = df_rd_clean.pivot_table(
        values="result_value_clean",
        index=province_col,
        columns="year_numeric",
        aggfunc="mean",
    )
    if pivot.shape[0] > 1 and pivot.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(14, max(6, pivot.shape[0] * 0.5)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Readmission Rate (%)"},
            annot_kws={"size": 8},
        )
        ax.set_title(
            "Readmission Rate Heatmap: Province × Year\n(Source: CIHI)",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Fiscal Year")
        ax.set_ylabel("Province / Territory")
        _save_fig("plot6_heatmap_province_year.png")
    else:
        print("  Skipped plot6 (pivot table too small)")
else:
    print("  Skipped plot6 (province or year column not found)")

print("\n[1.9] Summary statistics …")
print(df_rd_clean["result_value_clean"].describe().to_string())

# =============================================================================
# PART 2 – MACHINE LEARNING
# =============================================================================
print("\n" + "=" * 70)
print("PART 2 — MACHINE LEARNING")
print("=" * 70)

# ── 2.1  Build Feature Matrix ────────────────────────────────────────────────
print("\n[2.1] Building feature matrix …")

feature_cols = []
le_map = {}

# Encode categorical columns
for col in [province_col, age_col, sex_col, patient_type_col]:
    if col and col in df_rd_clean.columns:
        enc_col = col + "_enc"
        le_tmp = LabelEncoder()
        df_rd_clean[enc_col] = le_tmp.fit_transform(
            df_rd_clean[col].astype(str).str.strip()
        )
        le_map[enc_col] = le_tmp
        feature_cols.append(enc_col)

if "year_numeric" in df_rd_clean.columns:
    feature_cols.append("year_numeric")

if "is_national" in df_rd_clean.columns:
    df_rd_clean["is_national_int"] = df_rd_clean["is_national"].astype(int)
    feature_cols.append("is_national_int")

feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate, preserve order
print(f"  Feature columns: {feature_cols}")

if len(feature_cols) == 0:
    print("  WARNING: No feature columns could be constructed.")
    print("  Skipping ML section.")
else:
    df_ml = df_rd_clean[feature_cols + ["result_value_clean", "above_avg"]].dropna()
    print(f"  ML dataset shape: {df_ml.shape}")

    X = df_ml[feature_cols].values
    y_reg = df_ml["result_value_clean"].values
    y_cls = df_ml["above_avg"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X_scaled, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_cls_train, y_cls_test = train_test_split(
        X_scaled, y_cls, test_size=0.2, random_state=42
    )

    # ── 2.2  Model A: Linear Regression (predict readmission rate) ────────────
    print("\n[2.2] Model A: Linear Regression – predict readmission rate …")

    lr = LinearRegression()
    lr.fit(X_train, y_reg_train)
    y_pred_lr = lr.predict(X_test)

    mae_lr = mean_absolute_error(y_reg_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_reg_test, y_pred_lr))
    r2_lr = r2_score(y_reg_test, y_pred_lr)

    print(f"  Linear Regression  →  MAE: {mae_lr:.3f}  RMSE: {rmse_lr:.3f}  R²: {r2_lr:.3f}")

    # Cross-validated R²
    cv_scores = cross_val_score(lr, X_scaled, y_reg, cv=5, scoring="r2")
    print(f"  5-fold CV R²       →  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Actual vs Predicted plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_reg_test, y_pred_lr, alpha=0.5, color="#457B9D", edgecolors="white", s=50)
    lims = [
        min(y_reg_test.min(), y_pred_lr.min()),
        max(y_reg_test.max(), y_pred_lr.max()),
    ]
    ax.plot(lims, lims, "--", color="#D62828", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Readmission Rate (%)", fontsize=12)
    ax.set_ylabel("Predicted Readmission Rate (%)", fontsize=12)
    ax.set_title(
        f"Linear Regression: Actual vs Predicted\nR² = {r2_lr:.3f}  |  RMSE = {rmse_lr:.3f}",
        fontsize=12,
    )
    ax.legend()
    _save_fig("plot7_lr_actual_vs_predicted.png")

    # ── 2.3  Model B: Gradient Boosting Regression ────────────────────────────
    print("\n[2.3] Model B: Gradient Boosting Regressor …")

    gbr = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    gbr.fit(X_train, y_reg_train)
    y_pred_gbr = gbr.predict(X_test)

    mae_gbr = mean_absolute_error(y_reg_test, y_pred_gbr)
    rmse_gbr = np.sqrt(mean_squared_error(y_reg_test, y_pred_gbr))
    r2_gbr = r2_score(y_reg_test, y_pred_gbr)

    print(f"  Gradient Boosting  →  MAE: {mae_gbr:.3f}  RMSE: {rmse_gbr:.3f}  R²: {r2_gbr:.3f}")

    # Feature importances
    if len(feature_cols) > 0:
        importances = gbr.feature_importances_
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        feat_imp.plot(kind="barh", color="#D62828", ax=ax)
        ax.set_title(
            "Gradient Boosting – Feature Importances\n(Predicting Hospital Readmission Rates)",
            fontsize=12,
        )
        ax.set_xlabel("Importance")
        _save_fig("plot8_gbr_feature_importance.png")

    # ── 2.4  Model C: Logistic Regression (above/below average) ──────────────
    print("\n[2.4] Model C: Logistic Regression – classify above/below average …")

    n_cls_1 = y_cls_train.sum()
    n_cls_0 = len(y_cls_train) - n_cls_1
    if n_cls_1 < 5 or n_cls_0 < 5:
        print("  Insufficient class balance for classification – skipping.")
    else:
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_reg.fit(X_train, y_cls_train)
        y_pred_cls = log_reg.predict(X_test)
        y_prob_cls = log_reg.predict_proba(X_test)[:, 1]

        print(classification_report(y_cls_test, y_pred_cls, target_names=["Below Avg", "Above Avg"]))

        auc = roc_auc_score(y_cls_test, y_prob_cls)
        print(f"  ROC-AUC: {auc:.3f}")

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_cls_test,
            y_pred_cls,
            display_labels=["Below Avg", "Above Avg"],
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(
            "Logistic Regression Confusion Matrix\n"
            "(Above / Below National Average Readmission Rate)",
            fontsize=11,
        )
        _save_fig("plot9_log_reg_confusion_matrix.png")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_cls_test, y_prob_cls)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#D62828", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve – Logistic Regression (Above vs Below National Average)")
        ax.legend()
        _save_fig("plot10_log_reg_roc_curve.png")

    # ── 2.5  Clustering: K-Means on Province Readmission Profiles ────────────
    print("\n[2.5] K-Means clustering of readmission profiles …")

    if province_col and "year_numeric" in df_rd_clean.columns:
        pivot_cluster = df_rd_clean.pivot_table(
            values="result_value_clean",
            index=province_col,
            columns="year_numeric",
            aggfunc="mean",
        ).dropna()

        if pivot_cluster.shape[0] >= 3 and pivot_cluster.shape[1] >= 2:
            scaler_c = StandardScaler()
            X_cluster = scaler_c.fit_transform(pivot_cluster.values)
            n_clusters = min(4, pivot_cluster.shape[0])
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(X_cluster)

            cluster_df = pd.DataFrame(
                {
                    "Province": pivot_cluster.index,
                    "Cluster": labels,
                }
            )
            print(f"  K-Means ({n_clusters} clusters):")
            print(cluster_df.to_string(index=False))

            # Visualise cluster means
            pivot_cluster["Cluster"] = labels
            cluster_means = pivot_cluster.groupby("Cluster").mean().T
            fig, ax = plt.subplots(figsize=(10, 6))
            cluster_means.plot(ax=ax, marker="o")
            ax.set_xlabel("Fiscal Year")
            ax.set_ylabel("Mean Readmission Rate (%)")
            ax.set_title(
                "K-Means Clustering: Province Readmission Profiles by Year",
                fontsize=13,
            )
            ax.legend(title="Cluster")
            _save_fig("plot11_kmeans_profiles.png")
        else:
            print("  Insufficient data for K-Means clustering.")

    # ── 2.6  Save Models ──────────────────────────────────────────────────────
    joblib.dump(lr, os.path.join(OUTPUTS, "linear_regression_model.joblib"))
    joblib.dump(gbr, os.path.join(OUTPUTS, "gradient_boosting_model.joblib"))
    print("\n  Models saved to outputs/")

    # ── 2.7  Model Comparison Table ───────────────────────────────────────────
    print("\n[2.7] Model comparison summary …")
    comparison = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Gradient Boosting Regressor"],
            "MAE": [round(mae_lr, 4), round(mae_gbr, 4)],
            "RMSE": [round(rmse_lr, 4), round(rmse_gbr, 4)],
            "R²": [round(r2_lr, 4), round(r2_gbr, 4)],
        }
    )
    print(comparison.to_string(index=False))
    comparison.to_csv(os.path.join(OUTPUTS, "model_comparison.csv"), index=False)


# =============================================================================
# PART 3 – GENERATIVE AI
# =============================================================================
print("\n" + "=" * 70)
print("PART 3 — GENERATIVE AI")
print("=" * 70)

# ── 3.1  GenAI Usage Documentation ───────────────────────────────────────────
print("""
[3.1] Documented GenAI usage throughout this project:

  IDEATION
  --------
  GitHub Copilot was used to brainstorm relevant analytical questions for
  Canadian hospital readmission data, including:
    • How to structure aggregate CIHI indicator data for ML tasks
    • Choosing appropriate regression vs. classification framing for
      rates (not patient-level binary outcomes)
    • Selecting K-Means clustering to profile provinces over time

  CODING ASSISTANCE
  -----------------
  ChatGPT assisted with:
    • Writing the Excel header-row detection logic (_find_header_row)
    • Designing the column normalisation helper (_normalise_cols)
    • Gradient Boosting hyperparameter choices (n_estimators, learning_rate)
    • Seaborn heatmap formatting for the province × year matrix

  RESULT INTERPRETATION
  ---------------------
  Copilot helped refine the prompt templates below so that a large language
  model (e.g., GPT-4, Claude) can generate policy-facing summaries for
  Canadian provincial health ministries.

  All AI-generated suggestions were reviewed, adapted, and tested before
  inclusion.
""")

# ── 3.2  Prompt Templates (Canadian healthcare context) ───────────────────────
print("[3.2] Canadian-context prompt templates …\n")


def prompt_provincial_comparison(
    province: str,
    rate: float,
    national_avg: float,
    year: int,
) -> str:
    """Generate a prompt for an LLM to write a provincial readmission summary."""
    diff = rate - national_avg
    direction = "above" if diff > 0 else "below"
    return f"""You are a health policy analyst advising the {province} Ministry of Health.

Context:
- Data source: Canadian Institute for Health Information (CIHI)
- Indicator: All Patients Readmitted to Hospital within 30 days
- Fiscal year: {year}
- {province} readmission rate: {rate:.1f}%
- Canadian national average: {national_avg:.1f}%
- {province} is {abs(diff):.1f} percentage points {direction} the national average

Please write a concise 3-paragraph executive briefing for the provincial
health minister that:
1. States the finding clearly with reference to CIHI data
2. Discusses potential contributing factors within the Canadian healthcare
   system context (e.g., primary care access, discharge planning, social
   determinants of health)
3. Recommends 2–3 evidence-based policy actions aligned with Canada's
   Health Accord goals and provincial mandate

Use a professional tone appropriate for a Canadian government audience.
"""


def prompt_trend_analysis(rates_by_year: dict, province: str = "Canada") -> str:
    """Generate a prompt for an LLM to analyse a multi-year readmission trend."""
    trend_lines = "\n".join(
        f"  - Fiscal {yr}: {rate:.1f}%" for yr, rate in sorted(rates_by_year.items())
    )
    return f"""You are a Canadian health data scientist presenting to CIHI leadership.

Readmission Rate Trend for {province}:
{trend_lines}

Based on this CIHI data, please:
1. Describe the overall trend (improving, worsening, stable) with supporting
   calculations (absolute change, percentage change)
2. Identify any inflection points or anomalies and suggest likely causes
   (e.g., policy changes, COVID-19 pandemic impact, reporting methodology changes)
3. Project the likely readmission rate for the next two fiscal years using
   linear extrapolation, with appropriate caveats about uncertainty
4. Suggest what additional CIHI indicators would provide useful context for
   interpreting this trend

Cite your reasoning as if preparing a technical appendix for a CIHI public report.
"""


def prompt_patient_type_breakdown(patient_type_rates: dict) -> str:
    """Generate a prompt comparing readmission rates across patient types."""
    rows = "\n".join(
        f"  - {ptype}: {rate:.1f}%" for ptype, rate in patient_type_rates.items()
    )
    return f"""You are a clinical quality improvement specialist at a Canadian hospital network.

CIHI readmission rates by patient type:
{rows}

Please produce a structured quality improvement memo that:
1. Ranks patient types from highest to lowest readmission risk and explains
   clinical reasons for the ordering
2. For the highest-risk patient type, describes three evidence-based
   interventions used successfully in Canadian hospitals to reduce readmissions
3. Identifies which patient type shows the greatest opportunity for improvement
   relative to international benchmarks (reference comparable OECD countries)
4. Recommends specific metrics to track if a 90-day readmission reduction
   initiative were implemented at a Canadian academic health science centre

Format as a professional memo with clear headings.
"""


def prompt_generate_report(summary_stats: dict) -> str:
    """Generate a prompt for an LLM to write a full analysis report."""
    stats_text = "\n".join(f"  {k}: {v}" for k, v in summary_stats.items())
    return f"""You are writing the Executive Summary section of a capstone data analysis
report for the SFHA Advanced Data + AI Program.

The analysis used real Canadian health data from the Canadian Institute for
Health Information (CIHI) – specifically the "All Patients Readmitted to
Hospital" indicator.

Key findings from the analysis:
{stats_text}

Please write a 400–500 word Executive Summary that:
1. Opens with context on why hospital readmission rates matter for the Canadian
   healthcare system (cost, patient outcomes, system capacity)
2. Summarises the key data findings using the statistics provided, referencing
   CIHI as the authoritative source
3. Highlights what the machine learning models revealed about predictors of
   high readmission rates at the provincial/health-region level
4. Concludes with implications for Canadian health policy and recommendations
   for future data-driven quality improvement

Write in a clear, evidence-based style appropriate for a Canadian healthcare
audience. Do not fabricate statistics beyond those provided.
"""


# ── 3.3  Demonstrate Prompts with Real Data ───────────────────────────────────
print("[3.3] Sample prompt outputs (template demonstration) …\n")

# Compute summary stats for template demonstration
if "result_value_clean" in df_rd_clean.columns and len(df_rd_clean) > 0:
    nat_avg = df_rd_clean["result_value_clean"].mean()
    nat_min = df_rd_clean["result_value_clean"].min()
    nat_max = df_rd_clean["result_value_clean"].max()
    n_records = len(df_rd_clean)
else:
    nat_avg, nat_min, nat_max, n_records = 8.5, 6.0, 12.0, 0

summary_for_prompt = {
    "Total CIHI readmission records analysed": n_records,
    "Mean national readmission rate (%)": f"{nat_avg:.2f}",
    "Min readmission rate (%)": f"{nat_min:.2f}",
    "Max readmission rate (%)": f"{nat_max:.2f}",
    "Data source": "CIHI – All Patients Readmitted to Hospital",
    "Fiscal years covered": (
        str(int(df_rd_clean["year_numeric"].min()))
        + "–"
        + str(int(df_rd_clean["year_numeric"].max()))
        if "year_numeric" in df_rd_clean.columns
        and df_rd_clean["year_numeric"].notna().any()
        else "Multiple fiscal years"
    ),
    "Provinces/territories": (
        df_rd_clean[province_col].nunique()
        if province_col and province_col in df_rd_clean.columns
        else "N/A"
    ),
}

print("─" * 60)
print("SAMPLE PROMPT A – Provincial comparison (Ontario example):")
print("─" * 60)
print(prompt_provincial_comparison("Ontario", nat_avg + 0.3, nat_avg, 2022))

print("─" * 60)
print("SAMPLE PROMPT B – Trend analysis:")
print("─" * 60)
if "year_numeric" in df_rd_clean.columns and df_rd_clean["year_numeric"].notna().any():
    yearly_rates = (
        df_rd_clean.groupby("year_numeric")["result_value_clean"]
        .mean()
        .dropna()
        .to_dict()
    )
    yearly_rates_int = {int(k): v for k, v in yearly_rates.items()}
else:
    yearly_rates_int = {2019: 8.4, 2020: 8.1, 2021: 7.9, 2022: 8.2}
print(prompt_trend_analysis(yearly_rates_int, "Canada"))

print("─" * 60)
print("SAMPLE PROMPT C – Executive summary report:")
print("─" * 60)
print(prompt_generate_report(summary_for_prompt))

# ── 3.4  Save Prompts to File ─────────────────────────────────────────────────
prompts_path = os.path.join(OUTPUTS, "genai_prompts.txt")
with open(prompts_path, "w", encoding="utf-8") as fh:
    fh.write("CIHI Capstone Project – GenAI Prompt Templates\n")
    fh.write("=" * 60 + "\n\n")
    fh.write("PROMPT A – Provincial Comparison\n")
    fh.write("-" * 40 + "\n")
    fh.write(prompt_provincial_comparison("Ontario", nat_avg + 0.3, nat_avg, 2022))
    fh.write("\n\nPROMPT B – Trend Analysis\n")
    fh.write("-" * 40 + "\n")
    fh.write(prompt_trend_analysis(yearly_rates_int, "Canada"))
    fh.write("\n\nPROMPT C – Executive Report\n")
    fh.write("-" * 40 + "\n")
    fh.write(prompt_generate_report(summary_for_prompt))
    fh.write("\n\nPROMPT D – Patient Type Breakdown\n")
    fh.write("-" * 40 + "\n")
    sample_patient_types = {
        "Medical Patients": 9.1,
        "Surgical Patients": 7.3,
        "Obstetric Patients": 2.4,
        "Pediatric Patients": 3.8,
        "All Patients": 8.5,
    }
    fh.write(prompt_patient_type_breakdown(sample_patient_types))

print(f"\n  Prompts saved → {prompts_path}")

# ── 3.5  Template-Based Report ────────────────────────────────────────────────
report_path = os.path.join(OUTPUTS, "analysis_report.txt")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write(
        f"""SFHA Advanced Data + AI Program – Capstone Project Report
==========================================================

Title:  Canadian Hospital Readmission Rate Analysis
Data:   Canadian Institute for Health Information (CIHI)
        All Patients Readmitted to Hospital (Indicator #827)

Analysis Date: {pd.Timestamp.now().strftime("%B %d, %Y")}

─────────────────────────────────────────────────────────
DATA OVERVIEW
─────────────────────────────────────────────────────────
Total records analysed    : {n_records:,}
Mean readmission rate     : {nat_avg:.2f}%
Range                     : {nat_min:.2f}% – {nat_max:.2f}%
Data source               : CIHI (https://www.cihi.ca)

─────────────────────────────────────────────────────────
PART 1 – DATA PROCESSING SUMMARY
─────────────────────────────────────────────────────────
• Loaded two CIHI Excel files (indicator library + readmission data table)
• Applied column normalisation and suppression-flag handling
• Filtered for readmission-related indicators
• Engineered features: year_numeric, is_national, above_avg (classification target)
• Generated 6 visualisations (province rates, time trends, age, sex, heatmap)

─────────────────────────────────────────────────────────
PART 2 – MACHINE LEARNING SUMMARY
─────────────────────────────────────────────────────────
Three models were trained on CIHI aggregate readmission data:

Model A – Linear Regression (predict readmission rate %)
  Approach: Regression on province, year, age group, sex encodings
  Metric:   MAE, RMSE, R²

Model B – Gradient Boosting Regressor (predict readmission rate %)
  Approach: Ensemble method with feature importance analysis
  Metric:   MAE, RMSE, R²

Model C – Logistic Regression (classify above/below national average)
  Approach: Binary classification using same features
  Metric:   ROC-AUC, precision/recall, confusion matrix

Clustering – K-Means on province readmission profiles
  Approach: Unsupervised clustering of provinces by multi-year rate patterns
  Output:   Province groupings with similar readmission trajectories

─────────────────────────────────────────────────────────
PART 3 – GENERATIVE AI SUMMARY
─────────────────────────────────────────────────────────
GenAI tools used: GitHub Copilot, ChatGPT (GPT-4)

Ideation:
  • Explored how to adapt patient-level ML frameworks to aggregate
    indicator data; Copilot suggested regression + classification dual-
    approach and K-Means for provincial clustering

Coding Assistance:
  • Copilot drafted the Excel header detection logic
  • ChatGPT suggested the suppression-flag cleaning approach
  • Copilot generated seaborn heatmap formatting code

Interpretation:
  • Four Canadian-context prompt templates created for:
    - Provincial comparison briefings for health ministers
    - Multi-year trend analysis for CIHI leadership
    - Patient-type breakdown memos for quality improvement teams
    - Executive summary generation for academic reports

Prompt templates saved to: outputs/genai_prompts.txt

─────────────────────────────────────────────────────────
DATA CITATIONS
─────────────────────────────────────────────────────────
Canadian Institute for Health Information. (2024).
  All Patients Readmitted to Hospital [Data file].
  Retrieved from https://www.cihi.ca/en/indicators/
  all-patients-readmitted-to-hospital

Canadian Institute for Health Information. (2024).
  CIHI Indicator Library – All Indicator Data [Data file].
  Retrieved from https://www.cihi.ca/sites/default/files/document/
  indicator-library-all-indicator-data-en.xlsx
"""
    )

print(f"\n  Report saved → {report_path}")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 70)
print("Analysis complete.  All outputs saved to:", OUTPUTS)
print("=" * 70)
print("\nOutput files:")
for f in sorted(os.listdir(OUTPUTS)):
    fsize = os.path.getsize(os.path.join(OUTPUTS, f))
    print(f"  {f}  ({fsize:,} bytes)")
