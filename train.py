import logging
import os
import pickle
import re
import requests
import certifi
from typing import Any, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Configuration
DATA_PATH = "data/stock_data.csv"
DATA_URL = "https://raw.githubusercontent.com/yash612/Stockmarket-Sentiment-Dataset/master/stock_data.csv"
MODEL_PATH = "model.bin"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: Any) -> str:
    """Preprocesses a single stock market related text.

    Why: Convert to lowercase for normalization. Tickers (e.g., $AAPL) are replaced 
    with a generic token to preserve the financial context without overfitting 
    on specific symbols. URLs are removed as they don't contribute to sentiment.
    """
    text = str(text).lower()
    text = re.sub(r"\$\w+", "TICKER", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()


def download_data() -> bool:
    """Downloads the dataset if it's not present locally."""
    if os.path.exists(DATA_PATH):
        return True

    logger.info("Dataset not found. Attempting to download...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    try:
        response = requests.get(DATA_URL, verify=certifi.where(), timeout=30)
        response.raise_for_status()
        
        with open(DATA_PATH, "wb") as f:
            f.write(response.content)
            
        logger.info(f"Dataset successfully downloaded to {DATA_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def train() -> None:
    """Trains and tunes the sentiment analysis model.

    Why: Uses GridSearchCV to evaluate multiple models (LinearSVC and LogisticRegression) 
    and their hyperparameters (TF-IDF features, C parameter). Ensures selection of 
    the most accurate model for the financial context.
    """
    try:
        if not download_data():
            logger.error("Training aborted: Dataset is missing.")
            return

        logger.info("Loading and cleaning dataset...")
        df = pd.read_csv(DATA_PATH)
        x = df["Text"].apply(clean_text)
        df['Sentiment'] = df['Sentiment'].map({1: 1, -1: 0})
        y = df["Sentiment"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Modeling options for comparison
        pipelines = {
            "LinearSVC": Pipeline([
                ("tfidf", TfidfVectorizer(stop_words='english')),
                ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000))
            ]),
            "LogisticRegression": Pipeline([
                ("tfidf", TfidfVectorizer(stop_words='english')),
                ("clf", LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000))
            ])
        }

        # Parameter grids
        param_grids = {
            "LinearSVC": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__max_features": [5000, 10000],
                "clf__C": [0.1, 0.5, 1.0]
            },
            "LogisticRegression": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__max_features": [5000, 10000],
                "clf__C": [1.0, 10.0]
            }
        }

        best_score = 0
        best_model = None
        best_name = ""

        for name, pipeline in pipelines.items():
            logger.info(f"Tuning {name}...")
            grid_search = GridSearchCV(pipeline, param_grids[name], cv=3, scoring='f1_macro', n_jobs=-1)
            grid_search.fit(x_train, y_train)
            
            logger.info(f"{name} best score: {grid_search.best_score_:.4f}")
            logger.info(f"{name} best params: {grid_search.best_params_}")
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_name = name

        logger.info(f"### Final Selection: {best_name} with F1-Score {best_score:.4f} ###")

        # Final Evaluation
        y_pred = best_model.predict(x_test)
        logger.info("\n--- Final Performance Report ---")
        print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

        # Saving best artifact
        model_dir = os.path.dirname(MODEL_PATH)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        with open(MODEL_PATH, "wb") as f_out:
            pickle.dump(best_model, f_out)
        
        logger.info(f"Best model ({best_name}) exported successfully to {MODEL_PATH}")

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")


if __name__ == "__main__":
    train()
