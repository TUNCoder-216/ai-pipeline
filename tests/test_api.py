"""
tests/test_api.py — FastAPI endpoint tests
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

# ── We patch the model before importing the app ───────────────────────────────
import unittest.mock as mock

# Mock SentimentModel.get() so tests don't download a real model
MOCK_RESULT = {
    "text":       "I love this!",
    "label":      "POSITIVE",
    "score":      0.9987,
    "latency_ms": 12.3,
}

with mock.patch("src.model.SentimentModel.get") as mock_get:
    mock_model = mock.MagicMock()
    mock_model.predict.return_value = MOCK_RESULT
    mock_model.model_id = "test-model"
    mock_model.health.return_value = {"loaded": True, "model_id": "test-model"}
    mock_get.return_value = mock_model

    from src.api import app

client = TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_positive():
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= data["result"]["score"] <= 1.0


def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422   # Pydantic validation error


def test_predict_batch():
    mock_model.predict.return_value = [MOCK_RESULT, MOCK_RESULT]
    response = client.post("/predict/batch", json={"texts": ["Great!", "Terrible."]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_batch_too_large():
    texts = ["text"] * 101   # MAX_BATCH = 100
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 422
