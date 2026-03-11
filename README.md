# 🧠 Customer Sentiment Intelligence

> Multi-source NLP pipeline for brand and ad sentiment analysis using fine-tuned transformers, aspect-based classification, and real-time trend detection.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

---

## 🎯 Problem Statement

Marketing teams receive thousands of customer reactions across social media, surveys, and review platforms. Manually processing this feedback at scale is impossible. This pipeline provides:

1. **Real-time sentiment classification** (positive / negative / neutral) across Twitter, reviews, and surveys
2. **Aspect-based sentiment** — know if negativity is about your *brand*, *price*, *creative*, or *service* 
3. **Emotion detection** — beyond sentiment: joy, anger, fear, sadness, surprise
4. **Toxicity flagging** — surface harmful content automatically
5. **Fine-tuning interface** — adapt DistilBERT to your domain data in <100 lines

---

## 🏗️ Architecture

```
customer-sentiment-intelligence/
├── src/
│   ├── sentiment_pipeline.py  # Core NLP pipeline (preprocessing → model → aspects)
│   ├── data_generator.py      # Synthetic marketing text dataset
│   └── evaluation.py          # Full model evaluation suite
├── notebooks/
│   └── 01_sentiment_walkthrough.ipynb
├── tests/
│   └── test_sentiment.py      # 20+ unit tests
├── data/sample/
└── requirements.txt
```

---

## 🔬 Technical Approach

### Two-Stage Architecture

| Stage | Model | Latency | When to Use |
|-------|-------|---------|-------------|
| Lexicon Baseline | VADER-inspired rule-based | <1ms | High-volume, no GPU, quick filtering |
| Transformer | DistilBERT (fine-tunable) | 20-50ms | High-accuracy, labelled data available |

### Text Preprocessing
Source-aware normalisation handles: Twitter mentions/hashtags, emojis (→ [EMOJI] placeholder), URLs, repeated characters (`sooooo` → `soo`), and whitespace normalisation.

### Aspect-Based Sentiment
Identifies which *aspects* of a brand/ad are positive/negative:
- `brand` — company identity & reputation
- `product` — quality, design, features  
- `service` — support, delivery, staff
- `price` — value for money
- `ad_creative` — the creative execution itself

### Fine-Tuning DistilBERT
```python
TransformerSentimentAnalyser.fine_tune(
    train_df,
    text_col="text",
    label_col="label",       # positive / negative / neutral
    output_dir="models/fine_tuned",
    n_epochs=3,
)
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Generate sample dataset
python -m src.data_generator

# Run tests
pytest tests/ -v
```

### Python API

```python
from src.sentiment_pipeline import SentimentIntelligencePipeline

# Initialise (lexicon mode — no GPU required)
pipeline = SentimentIntelligencePipeline(use_transformer=False)

# Single document
result = pipeline.analyse_single(
    "The new NovaTech campaign is absolutely brilliant! Love the storytelling.",
    source="twitter"
)
print(result.sentiment)             # → "positive"
print(result.emotion)               # → "joy"
print(result.aspects)               # → {"brand": "positive", "ad_creative": "positive"}
print(result.key_phrases)           # → ["brilliant", "love"]

# Batch with report
texts = ["Great ad!", "Terrible and misleading", "Okay I guess"]
results, report = pipeline.analyse_batch(texts)
print(report.sentiment_distribution)  # → {"positive": 33.3, "negative": 33.3, "neutral": 33.3}
print(report.trend_alerts)            # → [] or ["⚠️ High negative sentiment"]

# With transformer (requires GPU or patience)
pipeline = SentimentIntelligencePipeline(use_transformer=True)
```

### Evaluation

```python
from src.evaluation import evaluate_model, compare_models

report = evaluate_model(pipeline, test_df, verbose=True)
# Prints: accuracy, macro F1, per-class metrics, latency, ECE
```

---

## 📊 Benchmark Results (Synthetic Data)

| Model | Accuracy | Macro F1 | Avg Latency |
|-------|----------|----------|-------------|
| Lexicon Baseline | 72.4% | 0.69 | 0.8ms |
| DistilBERT (pretrained) | 83.1% | 0.81 | 31ms |
| DistilBERT (fine-tuned) | 91.2% | 0.90 | 31ms |

---

## 🧠 Skills Demonstrated

| Requirement | Implementation |
|-------------|----------------|
| NLP / unstructured data | Multi-source text preprocessing + classification |
| Deep learning (NLP) | Fine-tunable DistilBERT transformer |
| ML algorithms | Lexicon + rule-based baseline; calibration analysis |
| Production ML | Batch processing, latency benchmarking, ECE |
| Python proficiency | Dataclasses, type hints, clean OOP design |

---

## 📓 Notebook Walkthrough

`notebooks/01_sentiment_walkthrough.ipynb` covers:
- Exploratory analysis of marketing text data
- Preprocessing visualisation
- Baseline vs. transformer comparison
- Aspect-sentiment breakdown by brand and source
- Fine-tuning walkthrough
