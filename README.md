# 🏥 Predicting 30-Day Hospital Patient Readmissions

**SFHA Advanced Data + AI Program — Week 8 Capstone Project**

A complete end-to-end healthcare analytics project demonstrating **Data Processing**, **Machine Learning**, and **Generative AI** to predict which hospital patients are at risk of being readmitted within 30 days of discharge.

---

## Problem Definition

### What is the problem?

Approximately **1 in 5 Medicare patients** is readmitted to hospital within 30 days of discharge. These preventable readmissions cost the U.S. healthcare system an estimated **\$26 billion annually** and are associated with significantly worse patient outcomes. The Centers for Medicare & Medicaid Services (CMS) penalises hospitals with high readmission rates, creating strong financial and quality-of-care incentives to address the problem.

### Why does it matter?

- **Financial impact**: Hospitals face CMS penalties of up to 3% of Medicare payments for excessive readmissions
- **Patient outcomes**: Readmissions are stressful and dangerous — patients re-exposed to hospital environments face risks of hospital-acquired infections and medication errors
- **Resource planning**: Accurate readmission predictions allow hospitals to proactively allocate transitional care resources (e.g., follow-up calls, home health visits, care coordinators)

### Who benefits?

| Stakeholder | Benefit |
|------------|---------|
| Patients | Targeted interventions reduce preventable readmissions |
| Hospitals | Reduced CMS penalties, better quality metrics |
| Care teams | Data-driven prioritisation of high-risk patients |
| Payers/Insurers | Lower total cost of care |

---

## Dataset Description

A **synthetic dataset of 5,000 hospital patients** is generated to simulate a realistic electronic health record (EHR) extract. The synthetic data preserves clinically realistic distributions and correlations.

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | String | Unique patient identifier |
| `age` | Integer | Patient age (18–90) |
| `gender` | Categorical | Male / Female |
| `admission_type` | Categorical | Emergency / Urgent / Elective |
| `diagnosis` | Categorical | Primary diagnosis (8 categories) |
| `num_procedures` | Integer | Number of procedures during stay |
| `num_medications` | Integer | Number of medications prescribed |
| `num_lab_tests` | Integer | Number of lab tests ordered |
| `length_of_stay` | Integer | Days in hospital |
| `num_previous_admissions` | Integer | Prior inpatient admissions |
| `has_diabetes` | Binary | Diabetes comorbidity flag |
| `has_hypertension` | Binary | Hypertension comorbidity flag |
| `has_heart_disease` | Binary | Heart disease comorbidity flag |
| `discharge_disposition` | Categorical | Home / SNF / Rehab / Home Health / AMA |
| `readmitted` | Binary | **Target**: 1 = readmitted within 30 days |

**Readmission rate**: ~22% (consistent with national averages)

---

## Project Structure

```
capstone-project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   ├── generate_synthetic_data.py     # Script to generate synthetic data
│   └── patient_readmission_data.csv   # Generated dataset (created on first run)
├── src/
│   ├── data_processing.py             # Cleansing, feature engineering, visualisation
│   ├── machine_learning.py            # Classification + clustering models
│   ├── generative_ai.py               # GenAI report generator + LLM prompts
│   └── main.py                        # Pipeline orchestrator
├── notebooks/
│   └── capstone_walkthrough.ipynb     # Interactive Jupyter walkthrough
└── outputs/                           # Generated on first run
    ├── plots/                         # PNG visualisation files
    ├── cleaned_data.csv               # Processed dataset
    ├── model_metrics.csv              # ML evaluation metrics
    ├── cluster_profiles.csv           # K-Means cluster summary
    └── analysis_report.md             # GenAI narrative report
```

---

## How to Install and Run

### Prerequisites
- Python 3.8 or higher

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python src/main.py
```

This will:
1. Generate the synthetic patient dataset (if not already present)
2. Process and clean the data, generate all visualisations
3. Train and evaluate ML models (classification + clustering)
4. Generate the Generative AI analysis report

All outputs are saved to the `outputs/` directory.

### 3. Interactive notebook walkthrough

```bash
jupyter notebook notebooks/capstone_walkthrough.ipynb
```

### 4. (Optional) Generate data independently

```bash
python data/generate_synthetic_data.py
```

---

## Core Concepts Demonstrated

### 1. 📊 Data Processing (`src/data_processing.py`)

**Where**: `src/data_processing.py`, Notebook Part 2

**What was done**:
- **Cleansing**: Detected and filled missing values (median/mode imputation), removed duplicates, capped outliers using IQR method
- **Manipulation / Feature Engineering**:
  - Age group binning (5 categories: 18–30, 31–45, 46–60, 61–75, 76+)
  - Comorbidity score (sum of 3 binary comorbidity flags)
  - Label encoding of all categorical variables
  - Min-Max scaling of numerical features
- **Visualisations** (5 plots saved to `outputs/plots/`):
  1. Readmission distribution (bar + pie chart)
  2. Feature correlation heatmap
  3. Age group distribution and readmission rates
  4. Length of stay analysis (histogram + box plots by diagnosis)
  5. Readmission rate by primary diagnosis

### 2. 🤖 Machine Learning (`src/machine_learning.py`)

**Where**: `src/machine_learning.py`, Notebook Part 3

**Classification (primary)**:
- 80/20 stratified train/test split
- Three models: **Logistic Regression**, **Random Forest**, **Gradient Boosting**
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Visualisations: Confusion matrices, ROC curves, feature importance plot

**Clustering (secondary)**:
- **K-Means clustering** (k=4) to identify patient risk segments
- Elbow method to determine optimal k
- PCA-based 2D cluster visualisation
- Cluster profiling (mean clinical metrics per segment)

**Typical results**:
| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~80% | ~0.62 | ~0.81 |
| Random Forest | ~83% | ~0.67 | ~0.86 |
| Gradient Boosting | ~84% | ~0.69 | ~0.88 |

### 3. 🧠 Generative AI (`src/generative_ai.py`)

**Where**: `src/generative_ai.py`, Notebook Part 4

**How GenAI was used in this project**:

1. **Ideation (ChatGPT)**: Used to brainstorm which clinical features have the strongest evidence base for 30-day readmission risk, shaping the synthetic data generation and feature engineering

2. **Code Generation (GitHub Copilot)**: Auto-completed repetitive patterns — the evaluation loop in `machine_learning.py`, seaborn plot boilerplate, and cluster profiling logic

3. **Debugging (ChatGPT)**: Resolved a `ValueError` with `LabelEncoder` on unseen categories by suggesting a consistent `fit_transform` pattern within the pipeline

4. **Report Drafting (ChatGPT)**: Narrative report templates were drafted with ChatGPT assistance, then edited for clinical accuracy

5. **System Integration (this module)**: Demonstrates how a production system would integrate with an LLM API:
   - `generate_analysis_prompts()` creates structured OpenAI-format prompts ready for any LLM API
   - `generate_report()` produces a full Markdown narrative report from ML results automatically

---

## Results Summary

- **Best model**: Gradient Boosting (highest F1 and ROC-AUC)
- **ROC-AUC ~0.88**: The model correctly ranks 88% of at-risk patient pairs — clinically meaningful performance for a screening tool
- **4 patient clusters** identified: low-risk young patients, moderate-risk middle-aged, high-risk elderly with multiple comorbidities, and very-high-risk frequent utilizers
- **Top predictors**: Comorbidity score, number of prior admissions, discharge disposition, and age

---

## Written Responses

### Question 1: Problem Definition

**What problem are you solving, why does it matter, and who benefits?**

I built a machine learning pipeline to predict **30-day hospital readmissions** — whether a patient discharged from hospital will return within 30 days. This is a well-documented, high-impact problem in healthcare analytics.

Readmissions matter because they are costly (\$26B/year in the US), harmful to patients, and often preventable with the right interventions. The key insight is that readmission risk is not uniform: some patients (elderly, multiple chronic conditions, discharged to home without support) are at significantly higher risk than others. A predictive model lets care teams focus limited transitional care resources on the patients who need them most.

Beneficiaries include patients (better outcomes, fewer hospitalizations), hospitals (reduced CMS penalties, improved quality scores), and the healthcare system overall (lower costs). The model can be embedded in discharge workflows: as a nurse completes discharge paperwork, the system automatically flags high-risk patients for care coordination follow-up.

### Question 2: Approach & Tools

**How did you approach the problem, what tools did you choose, and why?**

I chose Python as the primary language for its mature data science ecosystem and clinical analytics precedent. The core libraries were:

- **pandas/numpy**: Efficient data manipulation and numerical operations
- **scikit-learn**: Production-quality ML models with consistent API
- **matplotlib/seaborn**: Static visualisations appropriate for clinical reporting
- **Jupyter**: Interactive notebook for walkthrough documentation

For machine learning, I chose **classification** (the most direct fit for the binary readmission target) with three models of increasing complexity: Logistic Regression (interpretable baseline), Random Forest (robust ensemble), and Gradient Boosting (highest performance). I also added **K-Means clustering** as a secondary ML technique to provide actionable patient segmentation beyond binary prediction.

I could have used a neural network (e.g., PyTorch MLP), but tree-based ensemble methods are preferred in clinical settings because they handle tabular data well, require less preprocessing, and their feature importances are more interpretable to clinical stakeholders than neural network weights.

For Generative AI, I used **GitHub Copilot** throughout development for code completion and **ChatGPT** for ideation and debugging. I implemented a **template-based GenAI report generator** that works without paid API keys while demonstrating how the system would integrate with a real LLM API in production.

Alternatives considered: XGBoost (slightly better performance, but adds a dependency), SHAP for explainability (would add in a production version), and real datasets like MIMIC-III (used synthetic data to avoid privacy constraints and ensure reproducibility).

### Question 3: Reflection

**What worked well, what were the challenges, and what would you do differently?**

**What worked well**:
- The modular pipeline design (separate files for each stage) made it easy to test each component independently and will make future maintenance straightforward
- The synthetic data generation produced realistic enough distributions that the ML models achieved meaningful (not trivially perfect) performance
- Gradient Boosting consistently outperformed the other models, consistent with its reputation for tabular data
- GitHub Copilot significantly reduced the time spent on boilerplate code (evaluation loops, plot formatting, CSV saving) — probably saving 30–40% of development time

**Challenges**:
- Balancing the class imbalance (~22% positive rate) without overcomplicating the pipeline. I opted for stratified splits rather than SMOTE, which is appropriate for this dataset size but would need revisiting for production
- Making the synthetic data realistic enough that the feature importance rankings matched clinical expectations (comorbidity score and prior admissions should be top predictors) required several iterations on the probability distributions
- Structuring the GenAI component so it's genuinely useful (not just a gimmick) while working without paid API keys

**What I would change**:
- In production, I would validate on a real de-identified dataset (MIMIC-IV) and add SHAP explainability for individual predictions
- Add SMOTE or class-weight balancing to improve recall on the minority (readmitted) class
- Implement a proper ML experiment tracking system (MLflow) to compare model iterations
- Add a REST API endpoint so the model can be called from EHR systems in real-time
- Include social determinants of health features (zip code income level, insurance type, social support) which are strong real-world readmission predictors

---

## Technical Notes

- **No paid API keys required** — the project runs entirely on open-source tools
- All visualisations are saved as PNG files in `outputs/plots/`
- The pipeline is reproducible: fixed `random_seed=42` throughout
- Runs with a single command: `python src/main.py`

---

*Built as part of the **SFHA Advanced Data + AI Program** — Week 8 Capstone*

