"""
Machine Learning Module
========================
Trains and evaluates classification models to predict 30-day hospital
readmission, and applies K-Means clustering to identify patient risk segments.

Models implemented:
  Classification  – Logistic Regression, Random Forest, Gradient Boosting
  Clustering      – K-Means (elbow method + PCA visualisation)

GenAI Usage Note:
    GitHub Copilot assisted in writing the evaluation loop, suggesting the
    use of a dictionary-driven model registry pattern to keep the training
    code DRY. It also helped generate the cluster-profiling logic. All
    suggestions were reviewed and validated against scikit-learn documentation.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Features used for modelling
FEATURE_COLS = [
    "age",
    "num_procedures",
    "num_medications",
    "num_lab_tests",
    "length_of_stay",
    "num_previous_admissions",
    "has_diabetes",
    "has_hypertension",
    "has_heart_disease",
    "comorbidity_score",
    "gender_encoded",
    "admission_type_encoded",
    "diagnosis_encoded",
    "discharge_disposition_encoded",
]
TARGET_COL = "readmitted"


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

def _ensure_dirs() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _get_available_features(df: pd.DataFrame) -> list:
    """Return only the feature columns that actually exist in *df*."""
    return [c for c in FEATURE_COLS if c in df.columns]


# --------------------------------------------------------------------------- #
# 1. Classification
# --------------------------------------------------------------------------- #

def prepare_classification_data(df: pd.DataFrame):
    """
    Split data into train/test sets with stratification on the target.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame that includes engineered features.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, feature_names)
    """
    features = _get_available_features(df)
    X = df[features].fillna(0)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(
        f"[Classification] Train: {len(X_train)} rows | Test: {len(X_test)} rows"
        f" | Positive rate (train): {y_train.mean():.1%}"
    )
    return X_train, X_test, y_train, y_test, features


def train_classifiers(X_train, y_train) -> dict:
    """
    Train Logistic Regression, Random Forest, and Gradient Boosting classifiers.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like

    Returns
    -------
    dict
        Mapping of model name → fitted estimator.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"[Classification] Trained: {name}")
    return fitted


def evaluate_classifiers(
    fitted_models: dict,
    X_test,
    y_test,
    feature_names: list,
) -> pd.DataFrame:
    """
    Evaluate all classifiers and save confusion matrices, ROC curves, and
    feature importance plot.

    Parameters
    ----------
    fitted_models : dict
        Name → fitted model mapping.
    X_test : array-like
    y_test : array-like
    feature_names : list
        Names of the input features (used for importance plot).

    Returns
    -------
    pd.DataFrame
        Summary table of evaluation metrics for every model.
    """
    _ensure_dirs()
    results = []

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        }
        results.append(metrics)

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["Not Readmitted", "Readmitted"],
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix – {name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        cm_path = os.path.join(PLOTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png")
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()

        # ROC curve
        if y_prob is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
            ax.set_title(f"ROC Curve – {name}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            roc_path = os.path.join(PLOTS_DIR, f"roc_{name.lower().replace(' ', '_')}.png")
            plt.savefig(roc_path, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"[Evaluation] {name}: Acc={metrics['Accuracy']:.3f} | "
              f"F1={metrics['F1 Score']:.3f} | AUC={metrics['ROC-AUC']:.3f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Not Readmitted", "Readmitted"],
                                    zero_division=0))

    results_df = pd.DataFrame(results).set_index("Model")

    # Save metrics CSV
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    results_df.to_csv(metrics_path)
    print(f"[Evaluation] Metrics saved to {metrics_path}")

    # Feature importance (Random Forest)
    if "Random Forest" in fitted_models:
        _plot_feature_importance(fitted_models["Random Forest"], feature_names)

    return results_df


def _plot_feature_importance(rf_model, feature_names: list) -> None:
    """Plot and save feature importances from the Random Forest model."""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("viridis", len(sorted_features))
    plt.barh(sorted_features[::-1], sorted_importances[::-1], color=palette[::-1])
    plt.title("Feature Importance – Random Forest", fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "06_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {path}")


# --------------------------------------------------------------------------- #
# 2. Clustering (K-Means)
# --------------------------------------------------------------------------- #

def run_kmeans_clustering(df: pd.DataFrame, max_k: int = 10) -> dict:
    """
    Apply K-Means clustering to identify patient risk segments.

    Steps:
      1. Scale features
      2. Determine optimal k via the elbow method
      3. Fit final K-Means model (k=4 chosen based on elbow)
      4. Visualise clusters with PCA (2D)
      5. Profile each cluster

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame.
    max_k : int
        Maximum number of clusters to test in elbow method (default: 10).

    Returns
    -------
    dict
        Contains 'labels', 'cluster_profiles', 'pca_components', 'scaler'.
    """
    _ensure_dirs()
    features = _get_available_features(df)
    X = df[features].fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----- Elbow method -----
    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o", linewidth=2, color="#1976D2")
    plt.title("K-Means Elbow Method", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
    plt.xticks(list(k_range))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    elbow_path = os.path.join(PLOTS_DIR, "07_kmeans_elbow.png")
    plt.savefig(elbow_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Clustering] Elbow plot saved: {elbow_path}")

    # ----- Fit with k=4 -----
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df["cluster"] = labels
    print(f"[Clustering] K-Means fitted with k={optimal_k}. "
          f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")

    # ----- PCA visualisation -----
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        s=20,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(
        f"Patient Clusters (PCA 2D)\n"
        f"PC1={var_explained[0]:.1%} variance, PC2={var_explained[1]:.1%} variance",
        fontsize=12, fontweight="bold",
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    pca_path = os.path.join(PLOTS_DIR, "08_kmeans_clusters_pca.png")
    plt.savefig(pca_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Clustering] PCA cluster plot saved: {pca_path}")

    # ----- Cluster profiles -----
    profile_cols = [
        "age", "length_of_stay", "num_previous_admissions",
        "comorbidity_score", "num_medications", "readmitted",
    ]
    available_profile = [c for c in profile_cols if c in df.columns]
    cluster_profiles = df.groupby("cluster")[available_profile].mean().round(2)
    profile_path = os.path.join(OUTPUT_DIR, "cluster_profiles.csv")
    cluster_profiles.to_csv(profile_path)
    print(f"[Clustering] Cluster profiles saved: {profile_path}")
    print(cluster_profiles)

    return {
        "labels": labels,
        "cluster_profiles": cluster_profiles,
        "pca_components": X_pca,
        "scaler": scaler,
    }


# --------------------------------------------------------------------------- #
# Master run function
# --------------------------------------------------------------------------- #

def run_machine_learning(df: pd.DataFrame) -> dict:
    """
    Execute the full machine learning pipeline.

    Steps:
      1. Prepare train/test data
      2. Train classifiers
      3. Evaluate classifiers
      4. Run K-Means clustering

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame from the data processing module.

    Returns
    -------
    dict
        Contains 'fitted_models', 'metrics', 'clustering_results'.
    """
    print("\n" + "=" * 60)
    print("MACHINE LEARNING – CLASSIFICATION")
    print("=" * 60)
    X_train, X_test, y_train, y_test, features = prepare_classification_data(df)
    fitted_models = train_classifiers(X_train, y_train)
    metrics = evaluate_classifiers(fitted_models, X_test, y_test, features)

    print("\n" + "=" * 60)
    print("MACHINE LEARNING – CLUSTERING")
    print("=" * 60)
    clustering_results = run_kmeans_clustering(df)

    print("\n[Machine Learning] Pipeline complete.")
    return {
        "fitted_models": fitted_models,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "features": features,
        "clustering_results": clustering_results,
    }


if __name__ == "__main__":
    from data_processing import run_data_processing
    df = run_data_processing()
    run_machine_learning(df)
