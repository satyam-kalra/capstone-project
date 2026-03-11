# Written Response 2 – Approach & Tools

## SFHA Advanced Data + AI Program – Week 8 Capstone

---

### Overall Approach

The project follows an end-to-end data science workflow structured around
the three core requirements of the SFHA capstone:

```
Data Generation → Data Processing → Machine Learning → GenAI Integration
```

Because a real de-identified patient dataset was not available for this
project, a synthetic dataset of 1,000 patients was programmatically generated
with realistic probability distributions calibrated to published clinical
statistics (15–20% readmission rate, comorbidity prevalence, emergency
admission proportions). This approach is standard practice in healthcare AI
research for initial model development before obtaining ethics-board approval
to work with real EHR data.

---

### Tools and Methods

#### 1. Data Processing

| Tool | Purpose |
|------|---------|
| **pandas** | Data loading, cleansing (duplicates, missing values, type casting), and feature engineering |
| **NumPy** | Numerical operations and array manipulation |
| **matplotlib + seaborn** | Five publication-quality visualisations (distribution, rates by category, heatmap) |
| **scikit-learn LabelEncoder** | Encoding categorical variables for model input |

Key engineering decisions:
- **Age groups** (`Under 40`, `40–60`, `60–75`, `75+`) to capture non-linear
  age effects.
- **Polypharmacy flag** (≥10 medications): a clinically validated readmission
  risk factor supported by multiple peer-reviewed studies.
- **`med_proc_ratio`**: medication count divided by (lab procedures + 1) to
  capture medication complexity relative to clinical activity—suggested by
  ChatGPT during the ideation phase and validated against clinical literature.
- **`high_prior_admissions`**: binary flag for ≥2 previous admissions, the
  single strongest known predictor of readmission.

#### 2. Machine Learning

| Tool | Purpose |
|------|---------|
| **scikit-learn RandomForestClassifier** | Primary classification model |
| **scikit-learn LogisticRegression** | Baseline model for comparison |
| **scikit-learn metrics** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| **joblib** | Model serialisation (save/load) |

**Why Random Forest?**
- Handles non-linear interactions between features without manual feature
  crosses (e.g., emergency admission + high medications is more dangerous
  than either factor alone).
- Provides built-in feature importance ranking useful for clinical
  interpretation.
- Robust to outliers and skewed features common in clinical data.
- Strong baseline performance with minimal hyperparameter tuning.

**Why Logistic Regression?**
- Interpretable coefficients allow clinicians to understand exactly how each
  feature contributes to the risk score.
- Fast to train and useful as a calibrated probability estimator.
- Serves as a benchmark to demonstrate that the Random Forest's added
  complexity translates into measurable performance gain.

Both models were trained with `class_weight="balanced"` to compensate for
the approximately 4:1 class imbalance (not-readmitted vs. readmitted).

#### 3. Generative AI

| Tool | Usage |
|------|-------|
| **ChatGPT (GPT-4o)** | Ideation: feature engineering ideas, clinical context, intervention suggestions |
| **GitHub Copilot** | Coding assistance: boilerplate, matplotlib formatting, sklearn imports |
| **Template-based reporting** | System integration: structured narrative report generated from ML outputs |

The GenAI component was implemented in three ways matching the three allowed
rubric categories:
1. **Ideation**: ChatGPT was consulted for clinically meaningful features.
2. **Coding assistance**: Copilot accelerated development of boilerplate code.
3. **System integration**: A prompt-builder function (`build_patient_risk_prompt`)
   creates structured LLM prompts from patient data; the template-based
   `generate_analysis_report` function mimics LLM output generation.

---

### Alternatives Considered

| Alternative | Why Not Chosen |
|-------------|---------------|
| **Gradient Boosting (XGBoost/LightGBM)** | Stronger performance, but adds external dependency and complexity beyond the scope of a capstone demonstration |
| **Neural Network (PyTorch/Keras)** | Would demonstrate ML capability but requires more data and tuning than a 1,000-row synthetic dataset can support |
| **Real EHR dataset (e.g., MIMIC-III)** | Requires ethics approval and specific data-use agreements; synthetic data is appropriate for a course capstone |
| **Streamlit dashboard** | Would make a compelling demo but adds significant scope; the script + outputs approach is more accessible |
| **OpenAI API call (live)** | Requires a paid API key which not all reviewers will have; prompt-template approach demonstrates the same concepts without a credential barrier |

---

### Why These Choices Are Appropriate

The chosen stack—pandas, scikit-learn, matplotlib/seaborn—is the industry
standard for mid-level data science work in healthcare analytics. It is
well-documented, widely used, and requires no proprietary licenses. The
two-model comparison (interpretable baseline + stronger ensemble) mirrors
real clinical AI deployment workflows where regulatory bodies often require
an interpretable model alongside any black-box predictor. The documented
GenAI integration reflects how practitioners actually use these tools: not
as a replacement for domain knowledge, but as an accelerant for standard
tasks like code scaffolding, literature synthesis, and report drafting.
