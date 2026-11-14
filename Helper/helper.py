"""Reusable text preparation and review-summary utilities."""
import re
from pathlib import Path

import joblib
import pandas as pd


def clean_review(value: str) -> str:
    """Trim whitespace without altering the meaning of the review."""
    return re.sub(r"\s+", " ", str(value).strip())


def load_model(path: str | Path):
    """Load a trusted, locally trained scikit-learn pipeline."""
    return joblib.load(path)


def normalize_sentiment(value: str) -> str:
    labels = {"ايجابي": "إيجابي", "إيجابي": "إيجابي", "سلبى": "سلبي", "سلبي": "سلبي", "محايد": "محايد"}
    return labels.get(str(value).strip(), str(value).strip())


def sentiment_counts(df: pd.DataFrame) -> pd.Series:
    """Count all known sentiment classes, including classes with no predictions."""
    return df["Predicted_Sentiment"].value_counts().reindex(["إيجابي", "سلبي", "محايد"], fill_value=0)


def top_3_complaints(df: pd.DataFrame) -> tuple[pd.Series, int]:
    """Return top three topic counts among negative predictions and total negatives."""
    negatives = df.loc[df["Predicted_Sentiment"].eq("سلبي")]
    topic = negatives["Predicted_Attribute"].astype(str).str.strip()
    return topic[topic.ne("")].value_counts().head(3), len(negatives)
