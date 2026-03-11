"""
Synthetic marketing sentiment dataset generator.
Produces realistic ad/brand feedback across multiple channels.
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


# Realistic ad/brand feedback templates
POSITIVE_TEMPLATES = [
    "This {brand} ad is absolutely {adj_pos}! The {element} really resonated with me.",
    "Love the new {brand} campaign. The {element} is so {adj_pos} and {adj_pos2}.",
    "{brand} just released the most {adj_pos} commercial I've seen this year. {element} on point!",
    "Really {adj_pos} to see {brand} highlighting {topic}. Great job on the creative.",
    "The {element} in this {brand} ad is genuinely {adj_pos}. Sharing with everyone.",
    "Brilliant work from {brand}. The {element} is {adj_pos} and {adj_pos2}. 10/10.",
    "Just watched the new {brand} ad and it's {adj_pos}. The price point is also great.",
    "I'm actually impressed by this {brand} campaign. {element} is outstanding.",
]

NEGATIVE_TEMPLATES = [
    "Disappointed with {brand}'s new ad. The {element} is completely {adj_neg}.",
    "This {brand} campaign is {adj_neg} and {adj_neg2}. Who approved this?",
    "The new {brand} ad is {adj_neg}. {element} feels forced and {adj_neg2}.",
    "Really {adj_neg} to see {brand} doing this. The message is {adj_neg2} and misleading.",
    "{brand} just wasted my time with this {adj_neg} ad. The {element} is terrible.",
    "Not impressed with {brand}. This ad is {adj_neg} and the service is worse.",
    "The {brand} campaign fails completely. {element} is {adj_neg} and expensive.",
]

NEUTRAL_TEMPLATES = [
    "Saw the new {brand} ad. Not sure how I feel about the {element}.",
    "{brand} has a new campaign out. The {element} is okay I suppose.",
    "Noticed the {brand} ad. The {element} is interesting but nothing special.",
    "New ad from {brand}. Mixed feelings about the {element} honestly.",
]

BRANDS = ["NovaTech", "PureEssence", "MetroLife", "Zenith", "Luminary", "Apex"]
ELEMENTS = ["message", "visuals", "tagline", "music", "casting", "storytelling", "tone"]
TOPICS = ["sustainability", "diversity", "innovation", "community", "wellness"]
ADJ_POS = ["amazing", "brilliant", "creative", "inspiring", "engaging", "powerful", "refreshing"]
ADJ_NEG = ["boring", "misleading", "confusing", "offensive", "irrelevant", "terrible", "disappointing"]
SOURCES = ["twitter", "review", "survey", "social"]


def fill_template(template: str) -> str:
    return template.format(
        brand=random.choice(BRANDS),
        element=random.choice(ELEMENTS),
        topic=random.choice(TOPICS),
        adj_pos=random.choice(ADJ_POS),
        adj_pos2=random.choice([a for a in ADJ_POS]),
        adj_neg=random.choice(ADJ_NEG),
        adj_neg2=random.choice([a for a in ADJ_NEG]),
    )


def generate_sentiment_dataset(
    n_samples: int = 2000,
    pos_ratio: float = 0.45,
    neg_ratio: float = 0.30,
    seed: int = 42,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic marketing sentiment dataset.

    Args:
        n_samples: Total number of text samples
        pos_ratio: Fraction of positive samples
        neg_ratio: Fraction of negative samples
        seed: Random seed
        output_path: Optional CSV save path

    Returns:
        DataFrame with columns: text, label, source, brand,
                                created_at, char_length
    """
    random.seed(seed)
    np.random.seed(seed)

    n_pos = int(n_samples * pos_ratio)
    n_neg = int(n_samples * neg_ratio)
    n_neu = n_samples - n_pos - n_neg

    rows = []
    base_date = datetime(2024, 1, 1)

    for label, templates, n in [
        ("positive", POSITIVE_TEMPLATES, n_pos),
        ("negative", NEGATIVE_TEMPLATES, n_neg),
        ("neutral", NEUTRAL_TEMPLATES, n_neu),
    ]:
        for _ in range(n):
            text = fill_template(random.choice(templates))
            # Add some noise: occasional typos, caps, emoji-like
            if random.random() < 0.1:
                text = text.upper()
            if random.random() < 0.15:
                text += " 👏" if label == "positive" else " 😤" if label == "negative" else ""

            rows.append({
                "text": text,
                "label": label,
                "source": random.choice(SOURCES),
                "brand": random.choice(BRANDS),
                "created_at": base_date + timedelta(
                    days=random.randint(0, 365),
                    hours=random.randint(0, 23),
                ),
                "char_length": len(text),
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df)} samples to {output_path}")

    return df


if __name__ == "__main__":
    df = generate_sentiment_dataset(output_path="data/sample/sentiment_data.csv")
    print(df["label"].value_counts())
    print(df.head())
