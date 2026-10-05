# Jenkins LLMOps Delivery Platform

A production-oriented reference platform for validating, packaging, and deploying Hugging Face inference services to Kubernetes through Jenkins.

## What this project demonstrates

- Versioned inference APIs with readiness and liveness endpoints
- Reproducible container builds and non-root runtime security
- Jenkins quality gates for tests, image builds, and deployment manifests
- Helm-based Kubernetes releases with resource limits, autoscaling, and rollback support
- Prometheus-compatible latency, throughput, and error metrics

## Delivery flow

```text
commit -> unit tests -> container build -> Helm validation -> publish -> deploy
                                                        \-> health + SLO checks
```

The initial model is configurable through `MODEL_ID`. The default sentiment classifier keeps the first iteration inexpensive and verifiable; the same delivery path can later host automotive diagnostic and telemetry classifiers.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

```bash
curl http://localhost:8080/health/live
curl -X POST http://localhost:8080/v1/classify \
  -H 'content-type: application/json' \
  -d '{"text":"Battery temperature is above the recommended range."}'
```

## Roadmap

1. Add model-quality fixtures and latency budgets to the Jenkins quality gate.
2. Publish signed images and model metadata for every release.
3. Add canary promotion with automated rollback on SLO violations.
4. Replace the starter model with an automotive-domain classifier.

