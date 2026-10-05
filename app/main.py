import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app
from transformers import pipeline


MODEL_ID = os.getenv("MODEL_ID", "distilbert-base-uncased-finetuned-sst-2-english")
REQUESTS = Counter("inference_requests_total", "Inference requests", ["status"])
LATENCY = Histogram("inference_latency_seconds", "Inference request latency")

app = FastAPI(title="LLMOps Inference Service", version="0.1.0")
app.mount("/metrics", make_asgi_app())


class ClassificationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4_000)


class ClassificationResponse(BaseModel):
    label: str
    score: float
    model_id: str


@lru_cache(maxsize=1)
def classifier():
    return pipeline("text-classification", model=MODEL_ID)


@app.get("/health/live")
def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
def readiness():
    return {"status": "ready", "model_id": MODEL_ID}


@app.post("/v1/classify", response_model=ClassificationResponse)
def classify(request: ClassificationRequest):
    started = time.perf_counter()
    try:
        result = classifier()(request.text, truncation=True)[0]
        REQUESTS.labels(status="success").inc()
        return ClassificationResponse(
            label=result["label"], score=result["score"], model_id=MODEL_ID
        )
    except Exception as exc:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Inference unavailable") from exc
    finally:
        LATENCY.observe(time.perf_counter() - started)

