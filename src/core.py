import re

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

FILLER_WORDS = {"honestly", "fair", "overall", "experience", "ordered", "month", "bought", "gift", "second", "time", "ordering", "last", "ordered", "ones", "like", "lot"}
STOP_WORDS = list(ENGLISH_STOP_WORDS | FILLER_WORDS)

POSITIVE_WORDS = {"great", "excellent", "fast", "perfectly", "polite", "helpful", "solid", "neat", "worth", "accurate", "true", "easy", "quickly", "earlier", "cheaper"}
NEGATIVE_WORDS = {"broke", "cheap", "poor", "damaged", "rude", "refused", "never", "wrong", "faded", "overpriced", "small", "late", "waited", "rude", "not"}


def clean(text):
    text = text.lower()
    return re.sub(r"[^a-z\s]", " ", text)


def label_sentiment(rating):
    return (rating >= 4).astype(int)


def lexicon_baseline(texts):
    out = []
    for t in texts:
        words = set(clean(t).split())
        out.append(int(len(words & POSITIVE_WORDS) >= len(words & NEGATIVE_WORDS)))
    return np.array(out)


def sentiment_models(df, seed=0):
    y = label_sentiment(df["rating"])
    x_tr, x_te, y_tr, y_te = train_test_split(df["text"].map(clean), y, test_size=0.25, random_state=seed, stratify=y)
    vec = TfidfVectorizer(stop_words=STOP_WORDS, min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=1000, C=3.0).fit(vec.fit_transform(x_tr), y_tr)
    pred = clf.predict(vec.transform(x_te))
    base = lexicon_baseline(x_te)
    metrics = {
        "lexicon": {"accuracy": accuracy_score(y_te, base), "f1": f1_score(y_te, base)},
        "tfidf_logreg": {"accuracy": accuracy_score(y_te, pred), "f1": f1_score(y_te, pred)},
    }
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    words = {"negative": list(names[order[:10]]), "positive": list(names[order[-10:][::-1]])}
    return metrics, words


def topic_model(texts, n_topics=5, top_words=6, seed=0):
    vec = TfidfVectorizer(stop_words=STOP_WORDS, min_df=3, max_df=0.6)
    x = vec.fit_transform(texts.map(clean))
    nmf = NMF(n_components=n_topics, random_state=seed, init="nndsvda", max_iter=400)
    w = nmf.fit_transform(x)
    names = np.array(vec.get_feature_names_out())
    topics = [list(names[np.argsort(row)[::-1][:top_words]]) for row in nmf.components_]
    return topics, w


def topic_purity(weights, true_topics):
    """Share of single-topic reviews whose strongest NMF topic is the majority topic for that true label."""
    dom = weights.argmax(axis=1)
    single = true_topics.str.count(r"\|") == 0
    frame = pd.DataFrame({"dom": dom[single.to_numpy()], "true": true_topics[single].to_numpy()})
    return float(frame.groupby("true")["dom"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).mean())


def sentiment_by_topic(df, weights):
    dom = weights.argmax(axis=1)
    out = pd.DataFrame({"topic": dom, "positive": label_sentiment(df["rating"]).to_numpy()})
    return out.groupby("topic")["positive"].agg(["size", "mean"]).rename(columns={"mean": "positive_share"})
