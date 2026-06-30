"""Tests for evaluation metrics."""

from evaluation.metrics import compute_metrics, _normalize_prediction


def test_compute_metrics_perfect():
    y_true = ["positive", "negative", "positive", "negative"]
    y_pred = ["positive", "negative", "positive", "negative"]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_compute_metrics_partial():
    y_true = ["positive", "negative", "positive", "negative"]
    y_pred = ["positive", "positive", "positive", "negative"]
    metrics = compute_metrics(y_true, y_pred)
    assert 0.5 <= metrics["accuracy"] <= 1.0


def test_normalize_prediction():
    assert _normalize_prediction("The sentiment is positive.") == "positive"
    assert _normalize_prediction("This is negative overall.") == "negative"
    assert _normalize_prediction("ham") == "ham"
    assert _normalize_prediction("This is spam!") == "spam"


def test_imdb_sample_data():
    from datasets.imdb_loader import IMDBLoader
    df = IMDBLoader._get_sample_data()
    assert len(df) == 20
    assert set(df["label"].unique()) == {"positive", "negative"}


def test_spam_sample_data():
    from datasets.spam_loader import SpamLoader
    df = SpamLoader._get_sample_data()
    assert len(df) == 20
    assert set(df["label"].unique()) == {"ham", "spam"}
