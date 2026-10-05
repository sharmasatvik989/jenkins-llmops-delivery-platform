# Jenkins LLMOps Delivery Platform

A model-selection and deployment-planning platform for Hugging Face workloads. It helps teams identify a suitable model, understand the infrastructure required to operate it, and generate release configuration for Jenkins, Helm, and Kubernetes.

## Version 1

Version 1 is a live model advisor. It does not deploy a model or claim to operate a cluster.

1. Search and browse at least 15 models using live Hugging Face Hub metadata.
2. Organize models by category and supported task.
3. Inspect CPU or GPU suitability, model format, license, storage, estimated memory, library, revision, downloads, and gating status.
4. Recommend compute, resources, replicas, autoscaling, latency SLO, runtime, and model cache automatically.
5. Generate model ID and revision, container configuration, Helm values, Kubernetes resources, and Jenkins parameters.

Infrastructure recommendations are planning estimates derived from public model metadata, model size, task, format, and runtime. Production deployments must be benchmarked with representative traffic before approval.

## Architecture

```text
Browser -> FastAPI advisor -> Hugging Face Hub API
        -> compatibility and sizing rules
        -> Jenkins, Helm, Kubernetes and container configuration
```

The Hugging Face Hub is the model repository. This repository owns the advisory application and reusable delivery assets, not model weights.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn index:app --reload --port 8080
```

Open `http://localhost:8080`.

The included `app/main.py`, `Dockerfile`, Helm chart, and `Jenkinsfile` remain the reference inference workload and delivery path. Install `requirements-inference.txt` to run that workload.

The animated lifecycle uses [thinking-orbs](https://github.com/Jakubantalik/thinking-orbs), distributed under the MIT License.

## Deploy on Vercel

Import this repository into Vercel with the repository root (`./`) selected. Vercel detects the root `index.py` FastAPI entry point and installs `requirements.txt`. The browser-ready Thinking Orbs bundle is committed in `public/orbs.js`; rebuild it locally with `npm run build:orbs` whenever `src/orbs.jsx` changes.

- Framework preset: **FastAPI** (automatic detection is also supported)
- Root directory: **`./`**
- Environment variables: **none required for Version 1**
- Build and output settings: **use the repository settings**

The application reads public Hugging Face Hub metadata at request time. A deployment therefore needs normal outbound access to `huggingface.co`; no Hugging Face token is required for the public catalog used in Version 1.

## Roadmap

1. Benchmark generated recommendations against representative workloads.
2. Trigger a controlled Jenkins pipeline using generated parameters.
3. Report Kubernetes rollout health and inference SLOs.
4. Add canary promotion and automated rollback.
