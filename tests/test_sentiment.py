"""
Unit tests for Customer Sentiment Intelligence Pipeline
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sentiment_pipeline import (
    TextPreprocessor,
    LexiconSentimentAnalyser,
    SentimentIntelligencePipeline,
    SentimentResult,
    BatchSentimentReport,
)
from src.data_generator import generate_sentiment_dataset


# ── Preprocessor Tests ────────────────────────────────────────────────────────

class TestTextPreprocessor:
    def setup_method(self):
        self.pp = TextPreprocessor()

    def test_url_removal(self):
        result = self.pp.clean("Check out https://example.com for more info")
        assert "https" not in result["cleaned_text"]
        assert result["has_url"] is True

    def test_hashtag_extraction(self):
        result = self.pp.clean("Loving the #NewCampaign from the brand!")
        assert "NewCampaign" in result["hashtags"]

    def test_mention_handling_twitter(self):
        result = self.pp.clean("@BrandAccount this is amazing!", source="twitter")
        assert "@BrandAccount" not in result["cleaned_text"]
        assert "BrandAccount" in result["mentions"]

    def test_repeated_chars_normalised(self):
        result = self.pp.clean("This is sooooo amazing!")
        assert "soo" in result["cleaned_text"]
        # Should not have more than 2 consecutive same chars
        text = result["cleaned_text"]
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                pytest.fail(f"Repeated chars not normalised at position {i}: '{text[i:i+5]}'")

    def test_emoji_metadata(self):
        result = self.pp.clean("Great ad! 🎉🎊")
        assert result["emoji_count"] >= 1


# ── Lexicon Analyser Tests ────────────────────────────────────────────────────

class TestLexiconSentimentAnalyser:
    def setup_method(self):
        self.analyser = LexiconSentimentAnalyser()

    def test_positive_text(self):
        result = self.analyser.analyse("This is an excellent and amazing product!")
        assert result["sentiment"] == "positive"

    def test_negative_text(self):
        result = self.analyser.analyse("This is terrible and completely useless.")
        assert result["sentiment"] == "negative"

    def test_negation(self):
        result = self.analyser.analyse("This is not great at all.")
        # "not great" should reduce positive score
        assert result["score"] <= 0.3

    def test_confidence_range(self):
        texts = [
            "Amazing excellent wonderful!",
            "Okay I guess",
            "Terrible horrible dreadful",
        ]
        for text in texts:
            result = self.analyser.analyse(text)
            assert 0 <= result["confidence"] <= 1, f"Confidence out of range for: {text}"

    def test_aspect_detection(self):
        result = self.analyser.analyse("The brand reputation is great and the price is affordable")
        assert "brand" in result["aspects"]
        assert "price" in result["aspects"]

    def test_empty_text(self):
        result = self.analyser.analyse("")
        assert result["sentiment"] in ("positive", "negative", "neutral")


# ── Pipeline Tests ─────────────────────────────────────────────────────────────

class TestSentimentPipeline:
    def setup_method(self):
        self.pipeline = SentimentIntelligencePipeline(use_transformer=False)

    def test_single_analysis_returns_result(self):
        result = self.pipeline.analyse_single("I love this campaign!")
        assert isinstance(result, SentimentResult)

    def test_result_fields_populated(self):
        result = self.pipeline.analyse_single("Brilliant creative work from the brand.")
        assert result.sentiment in ("positive", "negative", "neutral")
        assert 0 <= result.sentiment_confidence <= 1
        assert result.emotion in ("joy", "anger", "sadness", "fear", "surprise", "neutral")
        assert 0 <= result.toxicity_score <= 1
        assert isinstance(result.key_phrases, list)

    def test_batch_analysis(self):
        texts = [
            "Amazing ad, love the storytelling",
            "Terrible campaign, very misleading",
            "Average creative, nothing special",
        ]
        results, report = self.pipeline.analyse_batch(texts)
        assert len(results) == 3
        assert isinstance(report, BatchSentimentReport)

    def test_batch_report_distribution_sums(self):
        texts = ["Great!"] * 5 + ["Bad!"] * 3 + ["Okay."] * 2
        _, report = self.pipeline.analyse_batch(texts)
        total = sum(report.sentiment_distribution.values())
        assert abs(total - 100) < 1.0  # should sum to ~100%

    def test_toxicity_high_for_toxic_text(self):
        result = self.pipeline.analyse_single(
            "This ad is hateful, offensive, and stupid."
        )
        assert result.toxicity_score > 0.1

    def test_processing_time_recorded(self):
        result = self.pipeline.analyse_single("Test text for timing.")
        assert result.processing_time_ms > 0

    def test_trend_alert_on_high_negative(self):
        texts = ["Terrible"] * 8 + ["Great"] * 2
        _, report = self.pipeline.analyse_batch(texts)
        # Should flag high negative sentiment
        if report.sentiment_distribution.get("negative", 0) > 40:
            assert len(report.trend_alerts) > 0


# ── Data Generator Tests ──────────────────────────────────────────────────────

class TestDataGenerator:
    def test_correct_n_samples(self):
        df = generate_sentiment_dataset(n_samples=100)
        assert len(df) == 100

    def test_label_distribution(self):
        df = generate_sentiment_dataset(n_samples=1000, pos_ratio=0.5, neg_ratio=0.3)
        pos_pct = (df["label"] == "positive").mean()
        assert abs(pos_pct - 0.5) < 0.05

    def test_required_columns(self):
        df = generate_sentiment_dataset(n_samples=50)
        for col in ["text", "label", "source", "brand", "created_at"]:
            assert col in df.columns

    def test_no_empty_texts(self):
        df = generate_sentiment_dataset(n_samples=200)
        assert (df["text"].str.len() > 0).all()

    def test_valid_labels(self):
        df = generate_sentiment_dataset(n_samples=100)
        assert set(df["label"].unique()).issubset({"positive", "negative", "neutral"})
