"""Optional TextBlob smoke coverage without downloaded corpora or network access."""

import socket

import pytest


@pytest.mark.parametrize(
    ("text", "direction"),
    [("I love this excellent result.", 1), ("I hate this terrible result.", -1)],
)
def test_textblob_sentiment_without_network_or_corpora(monkeypatch, text, direction):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Sentiment analysis must not access the network or download corpora")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setattr(socket, "getaddrinfo", forbid_network)

    nltk = pytest.importorskip("nltk")
    monkeypatch.setattr(nltk.data, "path", [])
    monkeypatch.setattr(nltk, "download", forbid_network)
    pytest.importorskip("textblob")

    from neural.analysis.sentiment import SentimentAnalyzer, SentimentEngine

    score = SentimentAnalyzer(engine=SentimentEngine.TEXTBLOB).analyze_text(text)

    assert "textblob" in score.metadata["engines_used"]
    assert score.engine_used is SentimentEngine.TEXTBLOB
    assert 0 < direction * score.overall_score <= 1
    assert 0 <= score.subjectivity <= 1
