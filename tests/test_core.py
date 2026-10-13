import numpy as np
import pytest

from src import core, data


@pytest.fixture(scope="module")
def reviews():
    return data.generate_reviews(n=1500, seed=4)


def test_clean_lowercases_and_strips_symbols():
    assert core.clean("Great!!! 10/10, FAST.").split() == ["great", "fast"]


def test_label_sentiment_threshold(reviews):
    labels = core.label_sentiment(reviews["rating"])
    assert set(np.unique(labels)) <= {0, 1}
    assert labels[reviews["rating"] >= 4].all()


def test_lexicon_baseline_on_obvious_cases():
    pred = core.lexicon_baseline(["great and fast", "broke and poor"])
    assert list(pred) == [1, 0]


def test_logistic_model_beats_lexicon(reviews):
    metrics, _ = core.sentiment_models(reviews)
    assert metrics["tfidf_logreg"]["accuracy"] > metrics["lexicon"]["accuracy"]
    assert metrics["tfidf_logreg"]["accuracy"] > 0.8


def test_topics_recover_planted_structure(reviews):
    topics, weights = core.topic_model(reviews["text"])
    assert len(topics) == 5
    assert core.topic_purity(weights, reviews["true_topics"]) > 0.6
