# Written Response 3 – Reflection

## SFHA Advanced Data + AI Program – Week 8 Capstone

---

### What Worked Well

**End-to-end pipeline design.** Structuring the project as a single runnable
Python script (`notebooks/capstone_analysis.py`) that flows logically from
data loading through processing, modelling, and reporting made the code easy
to follow, demonstrate in a video, and debug. Keeping it in one file (rather
than splitting across many modules) also reduced the cognitive overhead for
a reviewer exploring the repository for the first time.

**Synthetic data calibration.** Designing the data-generation script to
produce realistic class imbalance (~20% readmission rate) and clinically
correlated features (e.g., Emergency admissions having higher readmission
probability) meant the models trained on meaningful signal rather than random
noise. Both classifiers produced ROC-AUC scores well above 0.5, demonstrating
the models learned genuine patterns rather than overfitting to artefacts.

**GenAI integration documentation.** Explicitly documenting the three ChatGPT
and Copilot interactions in the code comments—with the actual prompts used and
how the outputs were incorporated—demonstrates transparent and responsible AI
use, a skill increasingly valued by healthcare employers.

**Visualisation quality.** Using seaborn's whitegrid theme, consistent colour
palettes (red for readmitted, blue for not readmitted), and annotated bar charts
produced presentation-quality plots suitable for clinical or executive audiences.

---

### What Did Not Work / Challenges

**Class imbalance.** With ~20% positive class, both models initially optimised
for accuracy by predicting "Not Readmitted" for most patients, yielding low
recall on the positive class. The `class_weight="balanced"` parameter improved
recall substantially but at the cost of some precision. In a real deployment
the threshold would be tuned based on the clinical cost tradeoff (false negative
= missed high-risk patient; false positive = unnecessary intervention).

**Limited dataset size.** A 1,000-row synthetic dataset is sufficient for a
capstone demonstration, but real-world healthcare models typically train on
tens of thousands of patient encounters to be reliable. The relatively small
dataset means confidence intervals around performance metrics are wide; a model
that scores AUC = 0.72 on 1,000 patients might score 0.65 or 0.79 on a
different sample from the same distribution.

**Feature correlation confounds.** Several engineered features are highly
correlated with their source columns (e.g., `high_prior_admissions` vs.
`num_previous_admissions`). This did not break the models but did inflate
feature importance estimates. SHAP values would provide a more robust
importance analysis for correlated features.

**No live LLM integration.** The GenAI reporting module generates structured
prompts rather than calling a live API, which means the "generated report"
is template-based rather than truly AI-generated. This was a deliberate
scope decision (no paid API key required), but the resulting output is less
impressive than a genuine GPT-generated narrative would be.

---

### What I Would Do Differently

1. **Use a real dataset**: The MIMIC-III or MIMIC-IV clinical database
   (publicly available with ethics training at physionet.org) would provide
   genuine predictive signal and a more rigorous model validation.
2. **SMOTE or cost-sensitive learning**: Address class imbalance more
   systematically using synthetic minority oversampling or a custom loss
   function rather than relying solely on class weights.
3. **SHAP explanations**: Add SHAP (SHapley Additive exPlanations) to provide
   individual-level feature attribution, which is essential for clinical
   trust and regulatory compliance in healthcare AI.
4. **Live LLM reporting**: Integrate a conditional OpenAI API call with a
   graceful fallback to the template-based report, so users with API access
   get a richer, context-aware narrative while others still get meaningful output.
5. **Cross-validation**: Replace a single train/test split with stratified
   k-fold cross-validation to get more reliable performance estimates.

---

### Lessons Learned

This project reinforced several important lessons about applied data science:

- **Data quality is the foundation**: No amount of model sophistication
  compensates for a poorly designed dataset. Time spent on feature engineering
  and data quality directly improved model performance.
- **Interpretability matters in healthcare**: A model that clinicians can
  explain and trust will be adopted; one they cannot will sit unused.
- **GenAI is a force multiplier, not a replacement**: Tools like ChatGPT and
  Copilot accelerated the project—particularly for literature review and
  boilerplate coding—but every AI suggestion required domain knowledge to
  evaluate and contextualise correctly.
- **Scope management is a skill**: A capstone project with unlimited scope
  expands infinitely. Defining clear boundaries (1,000 patients, two models,
  template-based GenAI) and delivering a polished, complete result within
  those boundaries is more valuable than an ambitious but incomplete project.
