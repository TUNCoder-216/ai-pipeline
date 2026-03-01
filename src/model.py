"""
=============================================================
  src/model.py — ML Model Wrapper
  Handles loading, caching, and inference for DistilBERT.
  Used by BOTH the FastAPI server AND the PySpark Pandas UDF.
=============================================================
"""

import os
import logging
import time
from typing import Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline,
)

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = os.getenv("MODEL_ID", "distilbert-base-uncased-finetuned-sst-2-english")
MODEL_CACHE = os.getenv("HF_HOME", "/model_cache")
BATCH_SIZE     = int(os.getenv("MODEL_BATCH_SIZE", "64"))
MAX_TOKEN_LEN  = int(os.getenv("MODEL_MAX_LENGTH", "512"))
DEVICE         = 0 if torch.cuda.is_available() else -1   # 0 = GPU, -1 = CPU

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "POSITIVE": "POSITIVE",
}


class SentimentModel:
    """
    Singleton-style wrapper around the HuggingFace sentiment pipeline.

    Why a class and not just a bare pipeline() call?
    ─────────────────────────────────────────────────
    1. We can lazy-load the model (only download/load on first use).
    2. We can expose metadata (model_id, device, load_time) for the /health endpoint.
    3. We can cleanly separate model logic from API logic.
    4. Unit tests can mock this class without touching HuggingFace at all.
    """

    _instance: "SentimentModel | None" = None   # module-level singleton cache

    def __init__(self):
        self._pipeline: Pipeline | None = None
        self.model_id    = MODEL_ID
        self.load_time_s: float | None = None
        self.device_str  = "cuda" if DEVICE == 0 else "cpu"

    # ── Singleton accessor ────────────────────────────────────────────────────
    @classmethod
    def get(cls) -> "SentimentModel":
        """Return the shared instance, creating and loading it if needed."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance

    # ── Model loading ─────────────────────────────────────────────────────────
    def load(self) -> None:
        """
        Download (first run) or load from cache (subsequent runs) the model.
        Explicitly loads tokenizer and model to prevent keyword leakage.
        """
        log.info(f"📦  Loading model '{self.model_id}' on {self.device_str} …")
        t0 = time.perf_counter()

        os.makedirs(MODEL_CACHE, exist_ok=True)

        # 1. Load the tokenizer and model explicitly.
        # This keeps 'cache_dir' out of the pipeline's internal 'params' list.
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)

        # 2. Initialize the pipeline using the objects we just created.
        # We pass truncation and max_length here; the pipeline handles them correctly
        # when they are passed this way.
        self._pipeline = pipeline(
            task="sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            truncation=True,
            max_length=MAX_TOKEN_LEN
        )

        self.load_time_s = round(time.perf_counter() - t0, 2)
        log.info(f"✅  Model loaded in {self.load_time_s}s")
    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, texts: Union[str, list[str]]) -> list[dict]:
        """
        Run inference on one or more texts.

        Returns a list of dicts, e.g.:
          [{"label": "POSITIVE", "score": 0.9987, "text": "Great product!"}]

        Args:
            texts: a single string OR a list of strings
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Normalise to list
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        t0 = time.perf_counter()
        raw = self._pipeline(texts)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        results = []
        for text, r in zip(texts, raw):
            results.append({
                "text":       text,
                "label":      LABEL_MAP.get(r["label"], r["label"]),
                "score":      round(float(r["score"]), 4),
                "latency_ms": latency_ms / len(texts),   # per-item estimate
            })

        log.info(f"🔮  Inference: {len(texts)} items in {latency_ms}ms")
        return results[0] if single else results

    # ── Health info ───────────────────────────────────────────────────────────
    def health(self) -> dict:
        return {
            "model_id":    self.model_id,
            "device":      self.device_str,
            "loaded":      self._pipeline is not None,
            "load_time_s": self.load_time_s,
            "batch_size":  BATCH_SIZE,
            "max_length":  MAX_TOKEN_LEN,
        }
