import numpy as np
import pandas as pd

TOPICS = {
    "delivery": {
        "pos": ["delivery arrived earlier than promised", "delivery was fast", "the delivery courier was polite", "delivery package came well wrapped"],
        "neg": ["delivery took two weeks", "the delivery courier left it outside", "delivery package arrived damaged", "delivery tracking never updated"],
    },
    "quality": {
        "pos": ["the quality of the material feels solid", "build quality is excellent", "quality stitching is neat", "quality looks exactly like the photos"],
        "neg": ["the quality of the material feels cheap", "quality is bad, it broke after a week", "poor build quality", "the quality is low and the color faded"],
    },
    "price": {
        "pos": ["great price for the money", "the price is lower than other shops", "the discount on the price was real", "price is worth every penny"],
        "neg": ["the price is too high for what you get", "found the price lower elsewhere", "the price went up after my order", "price is not worth the money"],
    },
    "support": {
        "pos": ["support answered within an hour", "support replaced the item quickly", "very helpful support staff", "support processed the refund fast"],
        "neg": ["support never replied", "support refused the refund", "rude support service", "waited days for the support answer"],
    },
    "size": {
        "pos": ["size fits perfectly", "the size chart was accurate", "true to size", "easy to assemble and the size fits"],
        "neg": ["size runs two sizes too small", "the size chart is wrong", "the size does not fit at all", "size is much smaller than expected"],
    },
}
OPENERS = ["", "Honestly, ", "To be fair, ", "Overall ", "My experience: "]
FILLERS = ["I ordered it last month.", "Bought it as a gift.", "Second time ordering here.", "", ""]


def generate_reviews(n=4000, seed=23):
    rng = np.random.default_rng(seed)
    topic_names = list(TOPICS)
    rows = []
    for i in range(n):
        positive = rng.random() < 0.62
        n_topics = rng.integers(1, 3)
        chosen = rng.choice(topic_names, n_topics, replace=False)
        parts = []
        for t in chosen:
            # a few reviews mix sentiments between topics
            pos_here = positive if rng.random() > 0.15 else not positive
            parts.append(rng.choice(TOPICS[t]["pos" if pos_here else "neg"]))
        text = rng.choice(OPENERS) + ", and ".join(parts) + ". " + rng.choice(FILLERS)
        rating = int(np.clip(round(rng.normal(4.4 if positive else 1.9, 0.7)), 1, 5))
        rows.append((i, text.strip(), rating, "|".join(sorted(chosen))))
    return pd.DataFrame(rows, columns=["review_id", "text", "rating", "true_topics"])
