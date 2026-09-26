"""
train.py
--------
Trains a Logistic Regression classifier on TF-IDF features extracted from
customer support ticket descriptions, evaluates it on the test split,
saves the confusion-matrix screenshot, and persists the model artefacts.

Usage
-----
    python src/train.py
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Allow importing sibling modules when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_and_preprocess  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")


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

    # ------------------------------------------------------------------
    # 3. Train Logistic Regression
    # ------------------------------------------------------------------
    print("Training Logistic Regression…")
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------
    # 5. Confusion matrix → screenshots/confusion_matrix.png
    # ------------------------------------------------------------------
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    cm_path = os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved -> {cm_path}")

    # ------------------------------------------------------------------
    # 6. Save model artefacts → model/
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(MODEL_DIR, "classifier.pkl"))
    print(f"Model artefacts saved -> {MODEL_DIR}/")


if __name__ == "__main__":
    train()
