import pytest
import unittest.mock as mock
from fastapi.testclient import TestClient

# -- Build the mock BEFORE importing the app --
mock_model = mock.MagicMock()
mock_model.model_id = "test-model"
mock_model.predict.return_value = {
    "text": "I love this!",
    "label": "POSITIVE",
    "score": 0.9987,
    "latency_ms": 12.3,
}
mock_model.health.return_value = {
    "loaded": True,
    "model_id": "test-model",
    "device": "cpu",
    "load_time_s": 1.0,
    "batch_size": 64,
    "max_length": 512,
}

# Patch BEFORE the app is imported so lifespan never calls the real model
with mock.patch("src.model.SentimentModel.get", return_value=mock_model):
    from src.api import app

client = TestClient(app, raise_server_exceptions=False)

# -- Tests --

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]

def test_health():
    with mock.patch("src.model.SentimentModel.get", return_value=mock_model):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_positive():
    with mock.patch("src.model.SentimentModel.get", return_value=mock_model):
        response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= data["result"]["score"] <= 1.0

def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_predict_batch():
    mock_model.predict.return_value = [
        {"text": "Great!", "label": "POSITIVE", "score": 0.99, "latency_ms": 10.0},
        {"text": "Terrible.", "label": "NEGATIVE", "score": 0.98, "latency_ms": 10.0},
    ]
    with mock.patch("src.model.SentimentModel.get", return_value=mock_model):
        response = client.post("/predict/batch", json={"texts": ["Great!", "Terrible."]})
    assert response.status_code == 200
    assert response.json()["count"] == 2

def test_batch_too_large():
    texts = ["text"] * 101
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 422
