"""
Customer Sentiment Intelligence Pipeline
==========================================
Fine-tuned transformer model for brand/ad sentiment analysis.
Handles multi-label classification: Sentiment + Emotion + Aspect.

Pipeline stages:
1. Data ingestion & preprocessing (multi-source: reviews, social, surveys)
2. Feature extraction (BERT embeddings + handcrafted features)
3. Fine-tuned DistilBERT classifier
4. Aspect-based sentiment (brand, product, service, price)
5. Trend detection + anomaly flagging
"""

import re
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    text: str
    sentiment: str                           # positive / negative / neutral
    sentiment_confidence: float
    emotion: str                             # joy / anger / fear / sadness / surprise
    emotion_confidence: float
    aspects: dict[str, str]                  # {"brand": "positive", "price": "negative"}
    toxicity_score: float
    key_phrases: list[str]
    processing_time_ms: float = 0.0


@dataclass
class BatchSentimentReport:
    n_documents: int
    sentiment_distribution: dict[str, float]
    emotion_distribution: dict[str, float]
    aspect_sentiment: dict[str, dict[str, float]]
    avg_toxicity: float
    top_positive_phrases: list[str]
    top_negative_phrases: list[str]
    trend_alerts: list[str] = field(default_factory=list)


# ── Text Preprocessing ────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    Multi-source text normalisation pipeline.
    Handles tweets, reviews, survey responses, and ad comments.
    """

    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F9FF"
        "\u2600-\u26FF\u2700-\u27BF"
        "]+",
        flags=re.UNICODE,
    )

    URL_PATTERN = re.compile(r"http\S+|www\.\S+")
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    REPEATED_CHARS = re.compile(r"(.)\1{3,}")

    def clean(self, text: str, source: str = "general") -> dict:
        """
        Clean and normalise text with source-aware rules.

        Args:
            text: Raw input text
            source: One of 'twitter', 'review', 'survey', 'general'

        Returns:
            Dict with cleaned text and extracted metadata
        """
        metadata = {
            "has_url": bool(self.URL_PATTERN.search(text)),
            "mentions": self.MENTION_PATTERN.findall(text),
            "hashtags": self.HASHTAG_PATTERN.findall(text),
            "emoji_count": len(self.EMOJI_PATTERN.findall(text)),
            "original_length": len(text),
        }

        # Replace emojis with text description placeholder
        cleaned = self.EMOJI_PATTERN.sub(" [EMOJI] ", text)

        # Source-specific handling
        if source in ("twitter", "social"):
            cleaned = self.MENTION_PATTERN.sub("[USER]", cleaned)
            cleaned = self.HASHTAG_PATTERN.sub(r"\1", cleaned)  # keep hashtag text

        cleaned = self.URL_PATTERN.sub("[URL]", cleaned)
        cleaned = self.REPEATED_CHARS.sub(r"\1\1", cleaned)  # loooove → love

        # Normalise whitespace
        cleaned = " ".join(cleaned.split())
        cleaned = cleaned.strip()

        metadata["cleaned_length"] = len(cleaned)
        metadata["cleaned_text"] = cleaned
        return metadata


# ── Lightweight Rule-Based Fallback (no GPU required) ────────────────────────

class LexiconSentimentAnalyser:
    """
    VADER-inspired lexicon-based sentiment analyser.
    Used as: (a) standalone baseline, (b) feature for transformer ensemble.
    """

    # Simplified sentiment lexicon (in production: load from VADER/SentiWordNet)
    POSITIVE_WORDS = {
        "excellent", "amazing", "outstanding", "fantastic", "wonderful", "great",
        "good", "love", "best", "perfect", "brilliant", "superb", "awesome",
        "happy", "satisfied", "recommend", "impressive", "innovative", "creative",
        "effective", "reliable", "trustworthy", "delightful", "engaging",
    }
    NEGATIVE_WORDS = {
        "terrible", "awful", "horrible", "dreadful", "worst", "bad", "hate",
        "disappointed", "poor", "useless", "broken", "boring", "offensive",
        "misleading", "confusing", "slow", "expensive", "annoying", "frustrating",
        "irrelevant", "waste", "scam", "fake", "wrong", "failed",
    }
    INTENSIFIERS = {"very", "extremely", "really", "absolutely", "incredibly", "super"}
    NEGATORS = {"not", "never", "no", "don't", "doesn't", "didn't", "won't", "cannot"}

    ASPECT_KEYWORDS = {
        "brand": ["brand", "company", "logo", "identity", "reputation", "name"],
        "product": ["product", "item", "quality", "feature", "design", "material"],
        "service": ["service", "support", "help", "staff", "team", "response", "delivery"],
        "price": ["price", "cost", "value", "expensive", "cheap", "affordable", "worth"],
        "ad_creative": ["ad", "advert", "commercial", "creative", "campaign", "message"],
    }

    def analyse(self, text: str) -> dict:
        """
        Score sentiment using lexicon + rule-based heuristics.

        Returns:
            Dict with sentiment label, score, aspects, and key phrases
        """
        tokens = text.lower().split()
        pos, neg = 0.0, 0.0
        multiplier = 1.0
        key_phrases = []

        for i, token in enumerate(tokens):
            # Look-behind for negation (within 3-word window)
            negated = any(
                tokens[max(0, i - 3) : i][j] in self.NEGATORS
                for j in range(min(3, i))
            )
            intensified = (
                tokens[i - 1] in self.INTENSIFIERS if i > 0 else False
            )
            weight = 1.5 if intensified else 1.0

            if token in self.POSITIVE_WORDS:
                if negated:
                    neg += weight
                else:
                    pos += weight
                    key_phrases.append(token)
            elif token in self.NEGATIVE_WORDS:
                if negated:
                    pos += weight * 0.5
                else:
                    neg += weight
                    key_phrases.append(token)

        total = pos + neg + 1e-10
        score = (pos - neg) / total  # [-1, 1]

        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"

        # Aspect detection
        text_lower = text.lower()
        aspects = {}
        for aspect, keywords in self.ASPECT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                aspects[aspect] = label  # simplified: same sentiment as overall

        return {
            "sentiment": label,
            "score": float(score),
            "confidence": min(abs(score) * 2, 1.0),
            "aspects": aspects,
            "key_phrases": list(set(key_phrases))[:5],
        }


# ── Transformer-Based Analyser ─────────────────────────────────────────────────

class TransformerSentimentAnalyser:
    """
    DistilBERT-based sentiment analyser.
    Uses HuggingFace transformers pipeline with fine-tuning support.

    In production, this would be fine-tuned on domain-specific
    marketing/ad data. Here we use a pretrained SST-2 checkpoint
    as a strong baseline and demonstrate the fine-tuning interface.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self._pipeline = None
        self._is_loaded = False

    def load(self):
        """Lazy-load the transformer pipeline (avoids cold start on import)."""
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                truncation=True,
                max_length=512,
                top_k=None,  # return all class scores
            )
            self._is_loaded = True
            print(f"✓ Loaded transformer: {self.model_name}")
        except Exception as e:
            print(f"⚠️  Could not load transformer ({e}). Falling back to lexicon analyser.")
            self._is_loaded = False

    def predict(self, text: str) -> dict:
        """Run inference on a single text."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        result = self._pipeline(text)[0]
        scores = {r["label"].lower(): r["score"] for r in result}

        sentiment = max(scores, key=scores.get)
        confidence = scores[sentiment]

        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "raw_scores": scores,
        }

    @classmethod
    def fine_tune(
        cls,
        train_df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        output_dir: str = "models/fine_tuned_sentiment",
        n_epochs: int = 3,
    ):
        """
        Fine-tune DistilBERT on domain-specific labeled data.

        Args:
            train_df: DataFrame with text and label columns
            text_col: Column name for input text
            label_col: Column name for sentiment labels (positive/negative/neutral)
            output_dir: Where to save the fine-tuned model
            n_epochs: Training epochs (3 is typically sufficient)

        Training data format:
            text                                    label
            "This ad is incredibly engaging"        positive
            "Misleading and boring campaign"        negative

        Usage:
            TransformerSentimentAnalyser.fine_tune(train_df)
            analyser = TransformerSentimentAnalyser("models/fine_tuned_sentiment")
            analyser.load()
        """
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
            )
            from datasets import Dataset
            import torch

            label2id = {"negative": 0, "neutral": 1, "positive": 2}
            id2label = {v: k for k, v in label2id.items()}

            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
            )

            def tokenize(batch):
                return tokenizer(
                    batch[text_col], truncation=True, padding=True, max_length=256
                )

            df = train_df.copy()
            df["labels"] = df[label_col].map(label2id)
            dataset = Dataset.from_pandas(df[[text_col, "labels"]])
            dataset = dataset.map(tokenize, batched=True)

            args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=n_epochs,
                per_device_train_batch_size=16,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                save_strategy="epoch",
                load_best_model_at_end=True,
                evaluation_strategy="epoch",
            )

            trainer = Trainer(model=model, args=args, train_dataset=dataset)
            trainer.train()
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            print(f"✓ Fine-tuned model saved to {output_dir}")
            return cls(output_dir)

        except ImportError:
            raise RuntimeError(
                "Fine-tuning requires: pip install transformers datasets torch"
            )


# ── Full Pipeline ─────────────────────────────────────────────────────────────

class SentimentIntelligencePipeline:
    """
    End-to-end sentiment analysis pipeline combining:
    - Text preprocessing
    - Lexicon-based baseline
    - Transformer-based deep model (optional)
    - Aspect extraction
    - Trend detection
    """

    def __init__(self, use_transformer: bool = False):
        self.preprocessor = TextPreprocessor()
        self.lexicon_analyser = LexiconSentimentAnalyser()
        self.transformer: Optional[TransformerSentimentAnalyser] = None

        if use_transformer:
            self.transformer = TransformerSentimentAnalyser()
            self.transformer.load()

    def analyse_single(self, text: str, source: str = "general") -> SentimentResult:
        """
        Analyse a single text document.

        Args:
            text: Input text (tweet, review, survey response)
            source: Data source for preprocessing rules

        Returns:
            SentimentResult dataclass
        """
        import time
        t0 = time.time()

        # Preprocess
        meta = self.preprocessor.clean(text, source=source)
        cleaned = meta["cleaned_text"]

        # Sentiment analysis
        if self.transformer and self.transformer._is_loaded:
            result = self.transformer.predict(cleaned)
            sentiment = result["sentiment"]
            sentiment_conf = result["confidence"]
        else:
            result = self.lexicon_analyser.analyse(cleaned)
            sentiment = result["sentiment"]
            sentiment_conf = result["confidence"]

        # Aspect analysis
        aspects = result.get("aspects", {})

        # Simple emotion classification (heuristic)
        emotion, emotion_conf = self._classify_emotion(cleaned)

        # Toxicity score (simple heuristic; in production: use Perspective API)
        toxicity = self._toxicity_score(cleaned)

        return SentimentResult(
            text=text,
            sentiment=sentiment,
            sentiment_confidence=round(sentiment_conf, 4),
            emotion=emotion,
            emotion_confidence=round(emotion_conf, 4),
            aspects=aspects,
            toxicity_score=round(toxicity, 4),
            key_phrases=result.get("key_phrases", []),
            processing_time_ms=round((time.time() - t0) * 1000, 2),
        )

    def analyse_batch(
        self, texts: list[str], sources: list[str] | None = None
    ) -> tuple[list[SentimentResult], BatchSentimentReport]:
        """
        Analyse a batch of documents and generate aggregate report.

        Returns:
            (list of SentimentResult, BatchSentimentReport)
        """
        if sources is None:
            sources = ["general"] * len(texts)

        results = [
            self.analyse_single(t, s) for t, s in zip(texts, sources)
        ]

        report = self._build_report(results)
        return results, report

    def _classify_emotion(self, text: str) -> tuple[str, float]:
        """Rule-based emotion classification (5 basic emotions)."""
        text_lower = text.lower()
        emotion_keywords = {
            "joy": ["happy", "love", "great", "amazing", "excited", "wonderful", "laugh"],
            "anger": ["angry", "furious", "outrage", "ridiculous", "disgusting", "hate"],
            "sadness": ["sad", "disappointed", "upset", "unhappy", "miss", "unfortunate"],
            "fear": ["worried", "concerned", "scary", "afraid", "anxious", "nervous"],
            "surprise": ["wow", "unexpected", "unbelievable", "shocked", "astonishing"],
        }
        scores = {
            e: sum(kw in text_lower for kw in kws)
            for e, kws in emotion_keywords.items()
        }
        best_emotion = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        conf = scores[best_emotion] / total
        if scores[best_emotion] == 0:
            return "neutral", 0.5
        return best_emotion, min(conf + 0.3, 1.0)

    def _toxicity_score(self, text: str) -> float:
        """Heuristic toxicity score [0, 1]. Replace with Perspective API in production."""
        toxic_keywords = [
            "hate", "kill", "stupid", "idiot", "moron", "awful", "disgusting",
            "offensive", "racist", "sexist", "violent",
        ]
        text_lower = text.lower()
        count = sum(1 for kw in toxic_keywords if kw in text_lower)
        return min(count / 5, 1.0)

    def _build_report(self, results: list[SentimentResult]) -> BatchSentimentReport:
        """Aggregate individual results into a batch report."""
        from collections import Counter

        sentiments = [r.sentiment for r in results]
        emotions = [r.emotion for r in results]

        sent_counts = Counter(sentiments)
        n = len(results)
        sent_dist = {k: round(v / n * 100, 1) for k, v in sent_counts.items()}

        emo_counts = Counter(emotions)
        emo_dist = {k: round(v / n * 100, 1) for k, v in emo_counts.items()}

        # Aspect sentiment aggregation
        aspect_sentiment: dict[str, dict[str, int]] = {}
        for r in results:
            for aspect, label in r.aspects.items():
                if aspect not in aspect_sentiment:
                    aspect_sentiment[aspect] = Counter()
                aspect_sentiment[aspect][label] += 1

        aspect_pct = {
            asp: {k: round(v / sum(counts.values()) * 100, 1) for k, v in counts.items()}
            for asp, counts in aspect_sentiment.items()
        }

        # Key phrases by sentiment
        pos_phrases = [p for r in results if r.sentiment == "positive" for p in r.key_phrases]
        neg_phrases = [p for r in results if r.sentiment == "negative" for p in r.key_phrases]

        # Trend alerts
        alerts = []
        neg_pct = sent_counts.get("negative", 0) / n
        if neg_pct > 0.4:
            alerts.append(f"⚠️  High negative sentiment: {neg_pct*100:.0f}% of documents")
        avg_tox = np.mean([r.toxicity_score for r in results])
        if avg_tox > 0.2:
            alerts.append(f"⚠️  Elevated toxicity detected: avg score {avg_tox:.2f}")

        return BatchSentimentReport(
            n_documents=n,
            sentiment_distribution=sent_dist,
            emotion_distribution=emo_dist,
            aspect_sentiment=aspect_pct,
            avg_toxicity=round(float(avg_tox), 4),
            top_positive_phrases=list(Counter(pos_phrases).most_common(10)),
            top_negative_phrases=list(Counter(neg_phrases).most_common(10)),
            trend_alerts=alerts,
        )
