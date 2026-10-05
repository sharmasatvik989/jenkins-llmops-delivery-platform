import asyncio
import math
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
HF_API = "https://huggingface.co/api/models"
SUPPORTED_TASKS = {"text-classification", "zero-shot-classification", "feature-extraction", "sentence-similarity", "summarization", "question-answering", "text-generation", "image-classification", "automatic-speech-recognition"}
CATALOG_TASKS = ["text-classification", "zero-shot-classification", "feature-extraction", "summarization", "question-answering", "text-generation", "image-classification", "automatic-speech-recognition"]
TASK_SUBCATEGORIES = {
    "text-classification": [("Sentiment", "sentiment"), ("Topic & intent", "topic classification"), ("Emotion", "emotion"), ("Safety & toxicity", "toxicity"), ("Financial language", "finbert"), ("Reranking", "reranker")],
    "zero-shot-classification": [("General zero-shot", "zero-shot"), ("Natural language inference", "mnli"), ("Multilingual", "multilingual zero-shot"), ("Topic routing", "topic zero-shot"), ("Intent detection", "intent zero-shot"), ("Safety routing", "safety zero-shot")],
    "feature-extraction": [("Semantic search", "sentence embeddings"), ("Retrieval", "retrieval embeddings"), ("Multilingual", "multilingual embeddings"), ("Code search", "code embeddings"), ("Compact", "small embeddings"), ("Reranking", "reranker")],
    "summarization": [("General", "summarization"), ("News", "news summarization"), ("Long documents", "long document summarization"), ("Dialogue", "dialogue summarization"), ("Scientific", "scientific summarization"), ("Multilingual", "multilingual summarization")],
    "question-answering": [("Extractive QA", "question answering"), ("Document QA", "document question answering"), ("Multilingual", "multilingual question answering"), ("Conversational", "conversational question answering"), ("Scientific", "scientific question answering"), ("Compact", "distilled question answering")],
    "text-generation": [("Instruction", "instruct"), ("Code", "code generation"), ("Small language models", "small language model"), ("Reasoning", "reasoning"), ("Multilingual", "multilingual generation"), ("Chat", "chat")],
    "image-classification": [("General vision", "image classification"), ("Objects", "object classification"), ("Medical", "medical image classification"), ("Documents", "document image classification"), ("Food", "food classification"), ("Satellite", "satellite image classification")],
    "automatic-speech-recognition": [("Whisper", "whisper"), ("Wav2Vec2", "wav2vec2"), ("Multilingual", "multilingual speech recognition"), ("English transcription", "english transcription"), ("Distilled", "distil whisper"), ("Streaming", "streaming speech recognition")],
}

FALLBACK_CATALOG = [
    ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "text-classification", "Sentiment"),
    ("SamLowe/roberta-base-go_emotions", "text-classification", "Emotion"),
    ("ProsusAI/finbert", "text-classification", "Financial language"),
    ("facebook/bart-large-mnli", "zero-shot-classification", "General zero-shot"),
    ("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "zero-shot-classification", "Multilingual"),
    ("sentence-transformers/all-MiniLM-L6-v2", "feature-extraction", "Semantic search"),
    ("BAAI/bge-small-en-v1.5", "feature-extraction", "Retrieval"),
    ("intfloat/multilingual-e5-small", "feature-extraction", "Multilingual"),
    ("facebook/bart-large-cnn", "summarization", "News"),
    ("google/pegasus-xsum", "summarization", "General"),
    ("deepset/roberta-base-squad2", "question-answering", "Extractive QA"),
    ("distilbert/distilbert-base-cased-distilled-squad", "question-answering", "Compact"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "text-generation", "Small language models"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "text-generation", "Instruction"),
    ("google/vit-base-patch16-224", "image-classification", "General vision"),
    ("microsoft/resnet-50", "image-classification", "Objects"),
    ("openai/whisper-small", "automatic-speech-recognition", "Whisper"),
    ("facebook/wav2vec2-base-960h", "automatic-speech-recognition", "Wav2Vec2"),
]

app = FastAPI(title="Jenkins LLMOps Model Advisor", version="1.0.0")
app.mount("/assets", StaticFiles(directory=ROOT / "public"), name="assets")

def human_bytes(value: int | float | None) -> str:
    if not value: return "Not published"
    size = float(value)
    for unit in ("B", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB": return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return "Not published"

def model_profile(model: dict[str, Any]) -> dict[str, Any]:
    tags, card, siblings = model.get("tags") or [], model.get("cardData") or {}, model.get("siblings") or []
    safetensors = model.get("safetensors") or {}
    params = safetensors.get("total") or (model.get("config") or {}).get("num_parameters")
    storage = model.get("usedStorage") or sum((item.get("size") or (item.get("lfs") or {}).get("size") or 0) for item in siblings)
    task = model.get("pipeline_tag") or "unknown"
    estimated_weights = max(storage or 0, (params * 2) if isinstance(params, (int, float)) else 0)
    generative = task in {"text-generation", "text2text-generation"}
    if estimated_weights >= 12 * 1024**3 or (generative and estimated_weights >= 4 * 1024**3):
        compute, gpu, replicas, maximum, latency, memory, cpu = "GPU", "NVIDIA L4 · 24 GB", 1, 4, 2500, max(16, math.ceil(estimated_weights / 1024**3 * 1.6)), "4"
    elif estimated_weights >= 2 * 1024**3 or generative:
        compute, gpu, replicas, maximum, latency, memory, cpu = "GPU", "NVIDIA T4 · 16 GB", 1, 4, 1800, max(8, math.ceil(estimated_weights / 1024**3 * 1.8)), "4"
    else:
        compute, gpu, replicas, maximum, latency, memory, cpu = "CPU", "Not required", 2, 8, 750, max(2, math.ceil(max(estimated_weights, 500_000_000) / 1024**3 * 2.5)), "2"
    names = [item.get("rfilename", "") for item in siblings]
    formats = []
    if any(name.endswith(".safetensors") for name in names) or "safetensors" in tags: formats.append("Safetensors")
    if any(name.endswith((".bin", ".pt", ".pth")) for name in names): formats.append("PyTorch")
    if any(name.endswith(".onnx") for name in names): formats.append("ONNX")
    if "gguf" in tags or any(name.endswith(".gguf") for name in names): formats.append("GGUF")
    library = model.get("library_name") or next((tag for tag in tags if tag in {"transformers", "sentence-transformers", "timm", "diffusers"}), "transformers")
    license_name = card.get("license") or next((tag.split("license:", 1)[1] for tag in tags if tag.startswith("license:")), "Not specified")
    gated = bool(model.get("gated"))
    notes = []
    if task not in SUPPORTED_TASKS: notes.append("Task needs a runtime adapter")
    if gated: notes.append("Requires Hugging Face access approval")
    if not formats: notes.append("Model format was not published in Hub metadata")
    return {"id": model.get("id"), "task": task, "subcategory": model.get("_advisor_subcategory") or "Popular", "library": library, "downloads": model.get("downloads") or 0, "likes": model.get("likes") or 0, "updated": model.get("lastModified"), "revision": model.get("sha") or "main", "license": license_name, "gated": gated, "formats": formats or ["Unspecified"], "parameters": params, "storage": human_bytes(storage), "estimated_weight_memory": human_bytes(estimated_weights), "compatible": task in SUPPORTED_TASKS and not gated, "compatibility_notes": notes, "recommendation": {"compute": compute, "gpu": gpu, "cpu": cpu, "memory": f"{memory}Gi", "replicas": replicas, "max_replicas": maximum, "autoscaling_cpu": 65, "latency_slo_ms": latency, "environment": "production", "runtime": "Hugging Face Transformers", "cache": f"{max(2, math.ceil((storage or estimated_weights or 1_000_000_000) / 1024**3 * 1.25))}Gi"}, "hugging_face_url": f"https://huggingface.co/{model.get('id')}"}

async def fetch_hub(params: dict[str, Any]) -> Any:
    try:
        async with httpx.AsyncClient(timeout=18, follow_redirects=True) as client:
            response = await client.get(HF_API, params=params, headers={"User-Agent": "jenkins-llmops-model-advisor/1.0"})
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return []

def fallback_models(task: str = "", limit: int = 18) -> list[dict[str, Any]]:
    rows = [row for row in FALLBACK_CATALOG if not task or row[1] == task]
    return [model_profile({
        "id": model_id,
        "pipeline_tag": model_task,
        "library_name": "transformers",
        "tags": ["transformers", "safetensors", "license:apache-2.0"],
        "downloads": 0,
        "sha": "main",
        "usedStorage": 1_000_000_000,
        "_advisor_subcategory": subcategory,
    }) for model_id, model_task, subcategory in rows[:limit]]

async def curated_models(task: str, limit: int, use_case: str = "") -> list[dict[str, Any]]:
    subcategories = TASK_SUBCATEGORIES[task]
    demand_words = set(use_case.lower().split()) - {"a", "an", "the", "model", "models", "text", "image", "audio", "speech", "classification", "recognition", "for", "to", "by"}
    if demand_words:
        matched = [item for item in subcategories if demand_words.intersection(f"{item[0]} {item[1]}".lower().split())]
        subcategories = matched + [item for item in subcategories if item not in matched]
    base = {"sort": "downloads", "direction": -1, "limit": 4, "full": "true", "config": "true", "pipeline_tag": task}
    groups = await asyncio.gather(*(fetch_hub({**base, "search": seed}) for _, seed in subcategories))
    selected, seen = [], set()
    for index in range(4):
        for (label, _), group in zip(subcategories, groups):
            if len(group) <= index: continue
            candidate = dict(group[index])
            if candidate.get("id") in seen: continue
            candidate["_advisor_subcategory"] = label
            selected.append(candidate)
            seen.add(candidate.get("id"))
            if len(selected) >= limit: return selected
    if len(selected) < limit:
        popular = await fetch_hub({**base, "limit": limit})
        for candidate in popular:
            if candidate.get("id") in seen: continue
            candidate = dict(candidate)
            candidate["_advisor_subcategory"] = "Popular"
            selected.append(candidate)
            seen.add(candidate.get("id"))
            if len(selected) >= limit: break
    return selected

@app.get("/")
def home(): return FileResponse(ROOT / "public" / "index.html")

@app.get("/api/health")
def health(): return {"status": "ok", "service": "model-advisor"}

@app.get("/api/models")
async def models(search: str = Query(default="", max_length=120), task: str = Query(default="", max_length=80), use_case: str = Query(default="", max_length=240), limit: int = Query(default=18, ge=15, le=40)):
    if not search.strip() and not task.strip():
        base = {"sort": "downloads", "direction": -1, "limit": 3, "full": "true", "config": "true"}
        groups = await asyncio.gather(*(fetch_hub({**base, "pipeline_tag": category}) for category in CATALOG_TASKS))
        balanced = [group[index] for index in range(3) for group in groups if len(group) > index][:limit]
        if not balanced:
            return {"models": fallback_models(limit=limit), "source": "Resilient catalog", "grouped": True}
        return {"models": [model_profile(model) for model in balanced], "source": "Hugging Face Hub", "grouped": True}
    if task.strip() in TASK_SUBCATEGORIES and not search.strip():
        data = await curated_models(task.strip(), limit, use_case)
        if not data:
            return {"models": fallback_models(task.strip(), limit), "source": "Resilient catalog", "grouped": True}
        return {"models": [model_profile(model) for model in data], "source": "Hugging Face Hub", "grouped": True}
    params: dict[str, Any] = {"sort": "downloads", "direction": -1, "limit": limit, "full": "true", "config": "true"}
    if search.strip(): params["search"] = search.strip()
    if task.strip(): params["pipeline_tag"] = task.strip()
    data = await fetch_hub(params)
    if not data:
        return {"models": fallback_models(task.strip(), limit), "source": "Resilient catalog"}
    return {"models": [model_profile(model) for model in data], "source": "Hugging Face Hub"}
