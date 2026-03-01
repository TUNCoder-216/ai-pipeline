"""
=============================================================
  src/api.py — FastAPI Inference Server
  Exposes the SentimentModel over HTTP.
  Endpoints:
    GET  /health       → service health + model info
    POST /predict      → single text sentiment
    POST /predict/batch → up to 100 texts at once
=============================================================
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.model import SentimentModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MAX_BATCH   = int(os.getenv("MAX_BATCH_SIZE", "100"))


# =============================================================================
# LIFESPAN — load the model ONCE when the server boots (not per request)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI's recommended way to run startup/shutdown logic.
    We load the model here so the first request isn't slow.
    """
    log.info("🚀  Server starting — loading model …")
    SentimentModel.get()          # warms up the singleton
    log.info("✅  Model ready. Server accepting requests.")
    yield
    log.info("🔌  Server shutting down.")


# =============================================================================
# APP DEFINITION
# =============================================================================

app = FastAPI(
    title="AI Sentiment Analysis API",
    description="Production-grade sentiment analysis powered by DistilBERT",
    version=APP_VERSION,
    lifespan=lifespan,
)

# ── CORS (allow any origin in dev; lock this down in production) ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST / RESPONSE SCHEMAS  (Pydantic validates inputs automatically)
# =============================================================================

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["I absolutely love this product!"],
    )

class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH,
        examples=[["Great product!", "Terrible quality."]],
    )

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls, v: list[str]) -> list[str]:
        if any(len(t.strip()) == 0 for t in v):
            raise ValueError("texts list cannot contain empty strings")
        return v

class SentimentResult(BaseModel):
    text:       str
    label:      str      # "POSITIVE" or "NEGATIVE"
    score:      float    # confidence 0.0 – 1.0
    latency_ms: float

class PredictResponse(BaseModel):
    result:     SentimentResult
    model_id:   str
    version:    str

class BatchPredictResponse(BaseModel):
    results:    list[SentimentResult]
    count:      int
    model_id:   str
    version:    str
    total_ms:   float


# =============================================================================
# MIDDLEWARE — log every request with timing
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"{request.method} {request.url.path} → {response.status_code} ({ms}ms)")
    response.headers["X-Response-Time-Ms"] = str(ms)
    return response


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/health", tags=["meta"])
def health():
    """
    Health check — used by Docker HEALTHCHECK, AWS ALB target group,
    and Kubernetes liveness/readiness probes.
    """
    model = SentimentModel.get()
    return {
        "status":  "ok",
        "version": APP_VERSION,
        "model":   model.health(),
    }


@app.get("/", tags=["meta"])
def root():
    return {"message": "AI Sentiment API is running 🚀", "docs": "/docs"}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    """
    Classify the sentiment of a single piece of text.

    Returns POSITIVE or NEGATIVE with a confidence score.
    """
    try:
        model  = SentimentModel.get()
        result = model.predict(req.text)
        return PredictResponse(
            result=SentimentResult(**result),
            model_id=model.model_id,
            version=APP_VERSION,
        )
    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(req: BatchPredictRequest):
    """
    Classify sentiment for up to 100 texts in one request.

    The model processes these as a vectorised batch — much faster
    than calling /predict 100 times individually.
    """
    if len(req.texts) > MAX_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"Maximum batch size is {MAX_BATCH}. Got {len(req.texts)}.",
        )

    try:
        t0     = time.perf_counter()
        model  = SentimentModel.get()
        results = model.predict(req.texts)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        return BatchPredictResponse(
            results=[SentimentResult(**r) for r in results],
            count=len(results),
            model_id=model.model_id,
            version=APP_VERSION,
            total_ms=total_ms,
        )
    except Exception as e:
        log.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RUN (local dev only — production uses Gunicorn in Docker)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,      # hot-reload on file changes (dev only!)
        log_level="info",
    )
