import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    df = data.generate_reviews()
    metrics, words = core.sentiment_models(df)
    topics, weights = core.topic_model(df["text"])
    purity = core.topic_purity(weights, df["true_topics"])
    by_topic = core.sentiment_by_topic(df, weights)
    by_topic["label"] = [", ".join(t[:3]) for t in topics]

    m = pd.DataFrame(metrics).T
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    m.plot.bar(ax=axes[0], color=["lightgrey", "black"], rot=0)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("Sentiment classification")
    by_topic.set_index("label")["positive_share"].plot.barh(ax=axes[1], color="grey")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Positive share by discovered topic")
    fig.tight_layout()
    fig.savefig(OUT / "text_mining.png", dpi=130)

    summary = {
        "reviews": int(len(df)),
        "positive_share": round(float(core.label_sentiment(df["rating"]).mean()), 3),
        "metrics": {k: {a: round(float(b), 3) for a, b in v.items()} for k, v in metrics.items()},
        "most_negative_terms": words["negative"][:6],
        "most_positive_terms": words["positive"][:6],
        "topics": topics,
        "topic_purity_single_topic_reviews": round(purity, 3),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
