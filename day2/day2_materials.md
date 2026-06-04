# Day 2 — MLOps in Production

> Hands-on companion: [day2_practice.ipynb](day2_practice.ipynb)

## Goals for Day 2

By the end of today, you will be able to:

1. **Package** a trained model and serve it behind a REST API.
2. Containerize the service for portable deployment.
3. Design a **CI/CD pipeline for ML** (testing data, code, and models).
4. Monitor a model in production and detect **data and concept drift**.
5. Apply industry best practices around **governance, security, and team structure**.

Recap from Day 1: [day1_materials.md](../day1/day1_materials.md).

---

## Module 5 — Packaging & Serving Models (09:00 – 10:30)

### 5.1 The packaging problem

A trained model in memory is useless in production. We need a **portable artifact** that another process (often on another machine) can load and call.

Common artifact formats:

| Format | Notes |
|--------|-------|
| `pickle` / `joblib` | Easy in Python; security risk if deserializing untrusted files; tied to library versions. |
| **ONNX** | Cross-framework, runtime-optimized inference. |
| **TorchScript** / SavedModel | Framework-native, production-grade. |
| **MLflow Model** | Wraps any of the above with metadata + signature. |

### 5.2 Model signature & schema

A model should declare what it accepts and returns:

```yaml
inputs:
  - {name: age,    type: long}
  - {name: income, type: double}
outputs:
  - {name: prob,   type: double}
```

This catches "the API sent a string where the model expected an int" **before** it crashes silently.

### 5.3 Serving patterns

| Pattern | When to use |
|---------|-------------|
| **Batch** (offline scoring, write to DB) | Daily recommendations, monthly risk scoring. |
| **Online / real-time REST/gRPC** | Latency-sensitive (fraud, search, ads). |
| **Streaming** | Continuous event scoring (Kafka, Flink). |
| **Embedded / on-device** | Mobile, IoT — use ONNX, TFLite, CoreML. |

### 5.4 A minimal online service (FastAPI)

**Why FastAPI in MLOps?**

**FastAPI** is a modern Python web framework used to wrap a trained model behind an HTTP API so other systems can request predictions over the network. In MLOps it serves as the "last mile" — the bridge between a model sitting in a file and a production application that needs real-time answers. You send a JSON request with input features, and the API returns the model's prediction. FastAPI is popular because it's fast, auto-generates API docs, and uses Pydantic for input validation (catching bad requests before they reach the model).

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")

class Features(BaseModel):
    mean_radius: float
    mean_texture: float
    # ... etc

@app.post("/predict")
def predict(f: Features):
    proba = model.predict_proba([[f.mean_radius, f.mean_texture]])[0, 1]
    return {"probability": float(proba)}

@app.get("/health")
def health():
    return {"status": "ok"}
```

Best practice endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness — am I alive? |
| `GET /ready` | Readiness — model loaded, deps reachable? |
| `POST /predict` | Inference |
| `GET /metrics` | Prometheus exposition format |

### 5.5 Containers

A `Dockerfile` makes the service portable:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
```

Then `docker build -t churn-svc:1.0 .` and `docker run -p 8000:8000 churn-svc:1.0`.

> **Best practice:** the **same** image runs in dev, staging and prod. Configuration comes from environment variables, not code.

➡ **Lab 4** in [day2_practice.ipynb](day2_practice.ipynb#Lab-4) — train, save, and serve a model with FastAPI.

---

## Module 6 — CI/CD for Machine Learning (10:45 – 12:15)

### 6.1 What's different from regular CI/CD?

Regular CI/CD validates **code**. ML CI/CD must additionally validate:

- **Data** (schema, distribution).
- **Model quality** (does the new model beat the current production model?).
- **Operational behaviour** (latency, payload size, memory).

### 6.2 Three pipelines, not one

A mature setup has three loosely-coupled pipelines:

| Pipeline | Trigger | Action |
|----------|---------|--------|
| **CI** (code) | Code change (push/PR) | Build + unit test + lint |
| **CT** (continuous training) | Data or code change | Retrain + evaluate + register if better |
| **CD** (deploy) | New registered model | Canary / staged rollout |

CI/CD/**CT** is the ML triad.

### 6.3 Tests you should have

| Layer | Example |
|-------|---------|
| Unit | `featurize_age(35) == 1` |
| Data | "no nulls in `user_id`", "label distribution within 5% of last week" |
| Model | "AUC on holdout ≥ 0.85", "AUC on protected group ≥ overall − 5%" |
| Behavioural | "increasing income increases predicted credit limit" (invariance / directional) |
| Integration | "POST /predict returns 200 with valid payload in <100 ms" |
| Smoke | hit `/health` after deploy |

### 6.4 GitHub Actions sketch

```yaml
name: ml-ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pytest -q
      - run: python -m src.pipeline.train --smoke
```

### 6.5 Safe deployment strategies

| Strategy | Description |
|----------|-------------|
| **Shadow / dark launch** | New model receives traffic but predictions are not used; compare to prod. |
| **Canary** | 1% → 10% → 100% with automated rollback on metric regression. |
| **A/B test** | Randomised split, business metric is the source of truth. |
| **Blue / green** | Two environments, switch the router. |

> **Best practice:** never replace a model "atomically" in production without a rollback plan.

➡ **Lab 5** in [day2_practice.ipynb](day2_practice.ipynb#Lab-5) — write model & data tests with `pytest`.

---

## Module 7 — Monitoring & Drift (13:15 – 14:45)

### 7.1 Why models decay

A trained model assumes the world keeps looking like the training set. It rarely does.

| Type of change | What changes | Example |
|----------------|--------------|---------|
| **Data drift** (covariate shift) | $P(X)$ | Users skew younger after a marketing campaign |
| **Concept drift** | $P(y \mid X)$ | Fraud patterns change after a new payment method |
| **Label drift** | $P(y)$ | Spam ratio drops from 30% to 5% |
| **Pipeline bugs** | nothing "real" | An ETL job started filling nulls with `0` |

### 7.2 What to monitor

1. **Service health** — uptime, latency, error rate (standard SRE).
2. **Data health** — input schema, null rates, feature distributions vs reference.
3. **Model health** — prediction distribution, confidence, segment-level metrics.
4. **Business KPIs** — conversion, revenue, complaints. The ultimate truth.

### 7.3 Detecting drift

**Numeric features:**
- **Population Stability Index (PSI)** — bin data, compare proportions:
  - PSI < 0.1: stable
  - PSI 0.1–0.25: warning
  - PSI > 0.25: alert
- **Kolmogorov–Smirnov** — non-parametric distribution test.
- **Wasserstein distance** — distance between distributions.

**Categorical features:**
- Chi-squared test
- JS divergence

Tools: **Evidently**, **NannyML**, **WhyLabs**, **Arize**, **Fiddler**.

### 7.4 The label problem

You usually can't compute model quality (AUC, etc.) in real time because **labels arrive late** (or never). Workarounds:

- **Proxy metrics**: prediction distribution shifts.
- **Delayed evaluation**: compute weekly with reconciled labels.
- **Active labelling**: route a small random % of predictions for human labelling.

### 7.5 Alerting that actually works

- Alert on **business symptoms**, not just statistical p-values.
- Every alert must have a **runbook**: how to diagnose, who owns it, how to roll back.
- Tune thresholds against **historical data** to avoid alert fatigue.

➡ **Lab 6** in [day2_practice.ipynb](day2_practice.ipynb#Lab-6) — generate drifted data and detect it (PSI + Evidently report).

---

## Module 8 — Industry Best Practices, Governance, Recap (15:00 – 16:30)

### 8.1 The MLOps stack — a buyer's map

| Concern | Open source | Managed |
|---------|-------------|---------|
| Experiment tracking | MLflow, Weights & Biases (free tier) | W&B, Neptune, Comet |
| Pipelines | Airflow, Prefect, Dagster, Kubeflow, Metaflow | Vertex AI Pipelines, Sagemaker Pipelines |
| Feature store | Feast | Tecton, Vertex FS, Sagemaker FS |
| Serving | BentoML, KServe, Triton, FastAPI | Sagemaker Endpoints, Vertex Endpoints |
| Monitoring | Evidently, NannyML | Arize, WhyLabs, Fiddler |
| End-to-end | ZenML, Metaflow, MLflow | Sagemaker, Vertex AI, Databricks |

> **Best practice:** start with the smallest stack that solves your real problems. Adopt new tools only when current pain justifies it.

### 8.2 Team structures

| Structure | Pros | Cons |
|-----------|------|------|
| **Embedded** (DS inside product team) | Fast iteration | Inconsistent practices |
| **Central platform team** | Consistent, reusable | Can become a bottleneck |
| **Hybrid** (central platform + embedded MLEs) | Balance of speed and consistency | More coordination needed |

### 8.3 Governance, risk & compliance

For regulated domains (finance, healthcare, EU AI Act) you must be able to answer:

| Question | Practice |
|----------|----------|
| What data trained this model? With what consent? | Model cards, datasheets, lineage |
| Who approved the deployment? | Approval workflow in model registry |
| How does the model behave on protected groups? | Bias/fairness evaluation in CI |
| How quickly can you turn it off? | Rollback plan, kill switch |

### 8.4 Security checklist

- Don't `pickle.load` untrusted artifacts. Prefer signed artifacts or safer formats (ONNX).
- Treat the model as a confidentiality risk: it can leak training data (membership inference, extraction).
- Validate **and rate-limit** prediction inputs (adversarial examples, abuse).
- Secrets via environment variables / secret managers — never in notebooks or git.
- Pin dependencies; scan for CVEs (e.g. `pip-audit`, Dependabot).

### 8.5 The non-negotiables (one slide)

| # | Principle | What it means |
|---|-----------|---------------|
| 1 | **Reproducibility** | Code + data + environment are versioned together |
| 2 | **Automation** | One command retrains, one command deploys |
| 3 | **Testing** | Code, data, and model quality all gate releases |
| 4 | **Observability** | You know when the model is misbehaving before users do |
| 5 | **Reversibility** | Every deployment can be rolled back quickly |
| 6 | **Ownership** | Every model has a named owner and a runbook |

### 8.6 Capstone discussion (group exercise)

Pick a real or imagined model in your organisation. As a group, sketch:

1. The **lifecycle diagram** (data → train → serve → monitor → retrain).
2. The **artifacts** that need versioning at each step.
3. The **tests** that gate promotion to production.
4. The **alerts** you would set up and the **runbook** for each.
5. The **maturity level** you are at and the next concrete step up.

➡ **Lab 7 (capstone)** in [day2_practice.ipynb](day2_practice.ipynb#Lab-7-Capstone) — connect the pieces into one mini end-to-end system.

---

## Course wrap-up

You now have a working mental model and concrete tooling experience for:

- The full ML lifecycle and where it breaks.
- Reproducible experiments with MLflow.
- Data validation and pipelines.
- Packaging, serving, and containerizing models.
- Testing and CI/CD adapted to ML.
- Monitoring and drift detection.
- Governance and team practices.

### Next steps

- **Read:** Huyen, *Designing Machine Learning Systems*; Kleppmann, *Designing Data-Intensive Applications*.
- **Build:** take one notebook from your day job and turn it into a reproducible, tracked, tested, served pipeline. That single exercise is worth more than ten more courses.
- **Stay current:** the [Made With ML](https://madewithml.com/) and [MLOps Community](https://mlops.community/) resources are excellent.
