"""
preprocessing.py
----------------
Loads the raw customer support tickets CSV, applies realistic noise
(label noise + keyword overlap), cleans the Ticket_Description text,
and returns train/test splits ready for TF-IDF vectorisation.
"""

import os
import re
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tickets.csv")


# ---------------------------------------------------------------------------
# Noise injection (matches notebook Task 3 requirement)
# ---------------------------------------------------------------------------
def make_dataset_realistic(
    df: pd.DataFrame,
    text_col: str = "Ticket_Description",
    target_col: str = "Issue_Category",
    noise_level: float = 0.15,
) -> pd.DataFrame:
    """
    Introduces realistic noise to break 100% determinism in the dataset.

    Steps
    -----
    1. Label noise  – randomly re-labels ~15 % of rows.
    2. Keyword overlap – appends a generic keyword to ~20 % of descriptions,
       simulating customer ambiguity.
    """
    df_modified = df.copy()
    np.random.seed(42)
    random.seed(42)

    # Step 1: label noise
    n_noise = int(len(df_modified) * noise_level)
    noise_indices = np.random.choice(df_modified.index, n_noise, replace=False)
    all_categories = df_modified[target_col].unique().tolist()

    for idx in noise_indices:
        current_cat = df_modified.loc[idx, target_col]
        wrong_cats = [c for c in all_categories if c != current_cat]
        df_modified.loc[idx, target_col] = random.choice(wrong_cats)

    # Step 2: keyword overlap
    overlap_keywords = ["account", "billing", "login", "payment", "error", "update", "broken"]
    overlap_indices = np.random.choice(
        df_modified.index, int(len(df_modified) * 0.20), replace=False
    )
    for idx in overlap_indices:
        random_word = random.choice(overlap_keywords)
        df_modified.loc[idx, text_col] = str(df_modified.loc[idx, text_col]) + f" {random_word}"

    return df_modified


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip boilerplate greetings, remove non-alpha characters."""
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove common synthetic greeting patterns
    text = re.sub(r"hi support,?", "", text)
    text = re.sub(r"hello,?", "", text)
    text = re.sub(r"dear team,?", "", text)

    # Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------
def load_and_preprocess(data_path: str = DATA_PATH, noise_level: float = 0.15):
    """
    Load, perturb, clean, and split the dataset.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.Series / pd.Series
        Cleaned text series and label series.
    """
    df = pd.read_csv(data_path)

    # Apply noise (must happen BEFORE the train/test split to avoid leakage)
    df = make_dataset_realistic(
        df,
        text_col="Ticket_Description",
        target_col="Issue_Category",
        noise_level=noise_level,
    )

    # Feature: cleaned ticket description; Target: issue category
    X_cleaned = df["Ticket_Description"].apply(clean_text)
    y = df["Issue_Category"]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X_cleaned, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print(f"Train size : {len(X_train)}")
    print(f"Test  size : {len(X_test)}")
    print(f"Classes    : {sorted(y_train.unique())}")
    print("\nSample cleaned text:")
    print(X_train.iloc[0])
