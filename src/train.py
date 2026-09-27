"""
train.py
--------
Benchmarks five classifiers on TF-IDF features extracted from customer
support ticket descriptions, saves a comparison chart, and persists all
trained model artefacts.

Usage
-----
    python src/train.py
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Allow importing sibling modules when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_and_preprocess  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")


MODEL_CONFIGS = {
    "Logistic Regression": (
        "logistic_regression.pkl",
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    ),
    "Multinomial Naive Bayes": (
        "multinomial_naive_bayes.pkl",
        MultinomialNB(alpha=0.5),
    ),
    "Linear SVM": ("linear_svm.pkl", LinearSVC(C=1.0, random_state=42)),
    "Random Forest": (
        "random_forest.pkl",
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    ),
    "Gradient Boosting": (
        "gradient_boosting.pkl",
        HistGradientBoostingClassifier(
            max_iter=30, learning_rate=0.1, max_depth=4, random_state=42
        ),
    ),
}


def train():
    # ------------------------------------------------------------------
    # 1. Load pre-processed data
    # ------------------------------------------------------------------
    print("Loading and preprocessing data…")
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # ------------------------------------------------------------------
    # 2. TF-IDF vectorisation  (fit ONLY on training data)
    # ------------------------------------------------------------------
    print("Vectorising text with TF-IDF…")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=1500,
        min_df=2,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Gradient Boosting does not accept sparse matrices. Keep sparse TF-IDF
    # for the other estimators and densify only this model's input.
    dense_train = None
    dense_test = None
    results = []
    trained_models = {}
    predictions = {}

    print("\nTraining and evaluating models...")
    for name, (_, model) in MODEL_CONFIGS.items():
        if name == "Gradient Boosting":
            if dense_train is None:
                dense_train = X_train_tfidf.toarray()
                dense_test = X_test_tfidf.toarray()
            train_features = dense_train
            test_features = dense_test
        else:
            train_features = X_train_tfidf
            test_features = X_test_tfidf

        print(f"- {name}")
        model.fit(train_features, y_train)
        y_pred = model.predict(test_features)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        results.append(
            {
                "Model": name,
                "Accuracy (%)": round(accuracy * 100, 2),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4),
            }
        )
        trained_models[name] = model
        predictions[name] = y_pred

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Accuracy (%)", ascending=False, kind="stable"
    ).reset_index(drop=True)
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    best_prediction = predictions[best_name]

    print("\n=== Performance Comparison ===")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print("\n=== Classification Report (Best Model) ===")
    print(classification_report(y_test, best_prediction))

    # ------------------------------------------------------------------
    # Comparison chart and best-model confusion matrix
    # ------------------------------------------------------------------
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    plt.figure(figsize=(11, 6))
    bars = plt.bar(
        results_df["Model"],
        results_df["Accuracy (%)"],
        color=["#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#dc2626"],
    )
    plt.title("Model Performance Comparison (Accuracy)", fontweight="bold")
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=15, ha="right")
    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.tight_layout()
    chart_path = os.path.join(SCREENSHOTS_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, best_prediction, labels=best_model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=best_model.classes_,
        yticklabels=best_model.classes_,
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    cm_path = os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Comparison chart saved -> {chart_path}")
    print(f"Confusion matrix saved -> {cm_path}")

    # ------------------------------------------------------------------
    # Save all model artefacts and expose the best model to the app.
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    for name, (filename, model) in MODEL_CONFIGS.items():
        joblib.dump(trained_models[name], os.path.join(MODEL_DIR, filename))
    joblib.dump(best_model, os.path.join(MODEL_DIR, "classifier.pkl"))
    results_path = os.path.join(MODEL_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Best classifier saved -> {os.path.join(MODEL_DIR, 'classifier.pkl')}")
    print(f"All model artefacts saved -> {MODEL_DIR}/")
    print(f"Metrics table saved -> {results_path}")


if __name__ == "__main__":
    train()
