# Canadian Hospital Readmission Rate Analysis

## SFHA Advanced Data + AI Program — Capstone Project

This project analyses **real Canadian hospital readmission data** from the
**Canadian Institute for Health Information (CIHI)** across three required capstone
components: Data Processing, Machine Learning, and Generative AI.

---

## Data Sources

All data comes from CIHI — Canada's authoritative source for health system performance information.

| File | Description | URL |
|------|-------------|-----|
| `indicator-library-all-indicator-data-en.xlsx` | CIHI Indicator Library — All Indicator Data | [Download](https://www.cihi.ca/sites/default/files/document/indicator-library-all-indicator-data-en.xlsx) |
| `827-all-patients-readmitted-to-hospital-data-table-en.xlsx` | 827 — All Patients Readmitted to Hospital Data Table | [Download](https://www.cihi.ca/sites/default/files/document/data-file/827-all-patients-readmitted-to-hospital-data-table-en.xlsx) |

**Data citation:**
> Canadian Institute for Health Information. (2024). *All Patients Readmitted to Hospital* [Data file].
> Retrieved from https://www.cihi.ca/en/indicators/all-patients-readmitted-to-hospital

---

## Project Structure

```
capstone-project/
├── data/
│   ├── download_cihi_data.py                               ← Download CIHI files
│   ├── indicator-library-all-indicator-data-en.xlsx        ← (downloaded)
│   └── 827-all-patients-readmitted-to-hospital-data-table-en.xlsx  ← (downloaded)
├── notebooks/
│   └── capstone_analysis.py                                ← Full analysis script
├── outputs/                                                ← Generated plots & reports
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the CIHI data

```bash
python data/download_cihi_data.py
```

This downloads both Excel files from CIHI into the `data/` directory and prints
basic metadata (sheet names, shape, preview rows).

### 3. Run the full analysis

```bash
python notebooks/capstone_analysis.py
```

All outputs (plots, cleaned data CSV, trained models, prompt templates, report)
are saved to the `outputs/` directory.

---

## What the Analysis Covers

### Part 1 — Data Processing
- Loads both CIHI Excel files with robust sheet/header detection
- Cleans suppressed values and missing data
- Filters indicator library for readmission-related indicators
- Engineers features: year, province encoding, above/below national average flag
- Generates 6 visualisations:
  - Readmission rates by province/territory
  - Rate trends over time
  - Rates by age group
  - Rates by sex
  - Rates by indicator/patient type
  - Province × Year heatmap

### Part 2 — Machine Learning
Three models trained on CIHI aggregate data:
- **Linear Regression** — predicts readmission rate (%)
- **Gradient Boosting Regressor** — predicts readmission rate with feature importance
- **Logistic Regression** — classifies provinces/regions as above/below national average
- **K-Means Clustering** — groups provinces by multi-year readmission profiles

### Part 3 — Generative AI
- Documents GitHub Copilot and ChatGPT usage (ideation, coding, interpretation)
- Four Canadian-context LLM prompt templates:
  - Provincial briefing for health ministers
  - Multi-year trend analysis for CIHI leadership
  - Patient-type quality improvement memo
  - Executive summary for academic reports
- Template-based analysis report referencing real CIHI findings

---

## Notes

- CIHI Excel files may contain multiple sheets, non-zero header rows, and merged cells.
  The analysis script handles all of these robustly.
- Suppressed values (marked with flags like "S", "SP") are replaced with `NaN`
  and excluded from calculations.
- The `outputs/` directory and downloaded data files are excluded from version control
  via `.gitignore`.

---

## Licence & Attribution

This project uses publicly available data from CIHI under their open data terms.
See https://www.cihi.ca/en/about-cihi/terms-of-use for details.
