"""
tests/test_api.py — FastAPI endpoint tests
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
import unittest.mock as mock

from src.api import app


MOCK_RESULT = {
    "text": "I love this!",
    "label": "POSITIVE",
    "score": 0.9987,
    "latency_ms": 12.3,
}


@pytest.fixture
def mock_model():
    with mock.patch("src.model.SentimentModel.get") as mock_get:
        model = mock.MagicMock()
        model.predict.return_value = MOCK_RESULT
        model.model_id = "test-model"
        model.health.return_value = {"loaded": True, "model_id": "test-model"}
        mock_get.return_value = model
        yield model


@pytest.fixture
def client(mock_model):
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_positive(client):
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= data["result"]["score"] <= 1.0


def test_predict_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_batch(client, mock_model):
    mock_model.predict.return_value = [MOCK_RESULT, MOCK_RESULT]
    response = client.post("/predict/batch", json={"texts": ["Great!", "Terrible."]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_batch_too_large(client):
    texts = ["text"] * 101
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 422
