"""
Main Pipeline Orchestrator
===========================
Runs the complete capstone project pipeline:
  1. Generate synthetic healthcare data
  2. Data processing (cleansing, feature engineering, visualisation)
  3. Machine learning (classification + clustering)
  4. Generative AI report generation

Usage:
    python src/main.py

GenAI Usage Note:
    ChatGPT was used to suggest the error-handling pattern used here —
    catching individual stage failures so that later stages can still run
    if earlier stages succeed.
"""

import sys
import os
import time

# Ensure src/ is on the path when running from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "patient_readmission_data.csv")


def _banner(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def stage_generate_data() -> bool:
    """Stage 1: Generate synthetic patient data if not already present."""
    _banner("STAGE 1: Synthetic Data Generation")
    if os.path.exists(DATA_PATH):
        print(f"[Stage 1] Data file already exists: {DATA_PATH}")
        print("[Stage 1] Skipping generation. Delete the file to regenerate.")
        return True
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, "data"))
        from generate_synthetic_data import generate_patient_data, save_dataset
        df = generate_patient_data(n_patients=5000)
        save_dataset(df, DATA_PATH)
        print("[Stage 1] Data generation complete.")
        return True
    except Exception as exc:
        print(f"[Stage 1] ERROR during data generation: {exc}")
        return False


def stage_data_processing():
    """Stage 2: Data cleansing, feature engineering, and visualisations."""
    _banner("STAGE 2: Data Processing")
    try:
        from data_processing import run_data_processing
        df = run_data_processing(DATA_PATH)
        print("[Stage 2] Data processing complete.")
        return df
    except Exception as exc:
        print(f"[Stage 2] ERROR during data processing: {exc}")
        return None


def stage_machine_learning(df):
    """Stage 3: Train ML models and run clustering."""
    _banner("STAGE 3: Machine Learning")
    try:
        from machine_learning import run_machine_learning
        results = run_machine_learning(df)
        print("[Stage 3] Machine learning complete.")
        return results
    except Exception as exc:
        print(f"[Stage 3] ERROR during machine learning: {exc}")
        return None


def stage_generative_ai(df, ml_results) -> bool:
    """Stage 4: Generate the GenAI narrative analysis report."""
    _banner("STAGE 4: Generative AI Report Generation")
    try:
        from generative_ai import run_generative_ai
        metrics_df = ml_results["metrics"]
        cluster_profiles = ml_results["clustering_results"]["cluster_profiles"]
        readmission_rate = df["readmitted"].mean()
        dataset_size = len(df)
        run_generative_ai(metrics_df, cluster_profiles, readmission_rate, dataset_size)
        print("[Stage 4] GenAI report generation complete.")
        return True
    except Exception as exc:
        print(f"[Stage 4] ERROR during GenAI reporting: {exc}")
        return False


def main() -> None:
    """Execute the full capstone pipeline end-to-end."""
    start_time = time.time()

    _banner("HEALTHCARE READMISSION PREDICTION — CAPSTONE PIPELINE")
    print("SFHA Advanced Data + AI Program")
    print("Predicting 30-Day Hospital Patient Readmissions\n")

    # Stage 1 – Data Generation
    if not stage_generate_data():
        print("\n[Pipeline] Aborting: data generation failed.")
        sys.exit(1)

    # Stage 2 – Data Processing
    df = stage_data_processing()
    if df is None:
        print("\n[Pipeline] Aborting: data processing failed.")
        sys.exit(1)

    # Stage 3 – Machine Learning
    ml_results = stage_machine_learning(df)
    if ml_results is None:
        print("\n[Pipeline] Aborting: machine learning failed.")
        sys.exit(1)

    # Stage 4 – Generative AI
    stage_generative_ai(df, ml_results)

    # Summary
    elapsed = time.time() - start_time
    _banner("PIPELINE COMPLETE")
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    print(f"Total runtime: {elapsed:.1f} seconds")
    print(f"\nOutputs saved to: {outputs_dir}/")
    print("  plots/                – all visualisation PNG files")
    print("  cleaned_data.csv      – processed dataset")
    print("  model_metrics.csv     – classification evaluation metrics")
    print("  cluster_profiles.csv  – K-Means cluster summary")
    print("  analysis_report.md    – GenAI narrative report")
    print("\nRun `jupyter notebook notebooks/capstone_walkthrough.ipynb` for")
    print("an interactive walkthrough of the full project.")


if __name__ == "__main__":
    main()
