# Day 1 — MLOps Foundations

> Hands-on companion: [day1_practice.ipynb](day1_practice.ipynb)

## Goals for Day 1

By the end of today, you will be able to:

1. Explain what MLOps is, **why** it exists, and how it differs from traditional DevOps.
2. Describe the end-to-end ML lifecycle and identify where things commonly break.
3. Make an ML project **reproducible** (environment, code, data, randomness).
4. Track experiments and models with **MLflow**.
5. Reason about **data versioning** and pipelines at a conceptual level.

---

## Module 1 — What is MLOps? (09:00 – 10:30)

### 1.1 Definition

> **MLOps** is the set of practices, tools, and culture that aim to deploy and maintain machine learning models in production reliably and efficiently.

It sits at the intersection of three disciplines:

```
        ┌──────────────┐
        │   ML / Data  │
        │   Science    │
        └──────┬───────┘
               │
   ┌───────────┴───────────┐
   │                       │
┌──┴────┐             ┌────┴────┐
│ DevOps│             │  Data   │
│       │             │ Eng.    │
└───────┘             └─────────┘
```

MLOps = **DevOps for ML**, plus everything DevOps doesn't cover (data, models, drift, experiments).

### 1.2 What is DevOps?

**DevOps** is a set of practices that combines software **Dev**elopment and IT **Op**erations. Its goal is to shorten the development lifecycle and deliver software reliably and continuously. Core ideas include:

- **Continuous Integration (CI)** — developers merge code frequently; each merge triggers automated builds and tests.
- **Continuous Delivery/Deployment (CD)** — code that passes CI is automatically packaged and deployed to production (or made ready for one-click deploy).
- **Infrastructure as Code** — servers and environments are defined in version-controlled config files (Terraform, Ansible), not set up manually.
- **Monitoring & feedback loops** — production systems are instrumented so teams learn quickly when something breaks.
- **Collaboration** — developers and operations engineers share ownership of the full lifecycle rather than throwing code "over the wall."

In short: DevOps ensures software goes from a developer's laptop to users quickly, safely, and repeatably. MLOps extends these ideas to the unique challenges of machine learning (data, models, drift).

### 1.3 Why MLOps? The "hidden technical debt" problem

The famous Sculley et al. paper (Google, 2015) showed that the ML model code is a **tiny fraction** of a real ML system. The rest is data collection, feature engineering, monitoring, configuration, serving infrastructure, etc.

Without MLOps:

- "It worked on my laptop" — model can't be reproduced.
- Models silently degrade as data changes (**drift**).
- No one knows which model version is in production.
- Retraining is a manual, multi-day, error-prone effort.
- Compliance / audit is impossible.

### 1.4 MLOps vs DevOps

### What is an "artifact"?

In software and ML, an **artifact** is any tangible output produced during the development process that you need to keep, version, or deploy. Examples:

- A compiled binary or Docker image (software artifact)
- A trained model file like `model.pkl` or `model.onnx` (ML artifact)
- A dataset snapshot (data artifact)
- A plot, report, or evaluation result (analysis artifact)

The **primary artifact** is the main deliverable of your workflow — in traditional software that's the application code/binary; in ML it's the code *plus* the trained model *plus* the data that produced it.

| Aspect | DevOps | MLOps |
|--------|--------|-------|
| Primary artifact | Code | Code **+ data + model** |
| Tests | Unit, integration | Plus **data validation, model quality** |
| Versioning | Source code | Source **+ data + model + experiment** |
| CI/CD | Build → Test → Deploy | Plus **train → evaluate → deploy → monitor** |
| Failure modes | Bugs, outages | Plus **drift, bias, silent quality decay** |
| Determinism | Mostly deterministic | Stochastic (seeds, hardware, data) |

### 1.5 The ML Lifecycle

```
   ┌──────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────┐   ┌──────────┐
   │   Data   │──▶│ Features │──▶│   Train /   │──▶│  Deploy  │──▶│ Monitor  │
   │ (collect,│   │          │   │  Evaluate   │   │  (serve) │   │ (drift,  │
   │  label)  │   │          │   │             │   │          │   │  perf)   │
   └──────────┘   └──────────┘   └─────────────┘   └──────────┘   └────┬─────┘
        ▲                                                              │
        └──────────────────────── retrain / iterate ───────────────────┘
```

Discussion: **Where in this loop have your past projects broken down?**

### 1.6 MLOps Maturity Levels (Google)

| Level | Description | When appropriate |
|-------|-------------|------------------|
| **Level 0** | Manual: scripts, notebooks, manual deploys | Proof of concept, early exploration |
| **Level 1** | ML pipeline automation: automated training, reproducible runs | Models with regular retraining needs |
| **Level 2** | CI/CD pipeline automation: code/data/model changes trigger automated retrain + deploy | High-velocity teams, many models |

**Key insight:** You don't always need Level 2. Pick the lowest level that solves your business pain.

---

## Module 2 — Reproducibility & Environments (10:45 – 12:15)

### 2.1 The four pillars of reproducibility

| Pillar | Tool examples |
|--------|---------------|
| Code | `git`, code review |
| Environment | `venv`, `conda`, `pip-tools`, Docker |
| Data | DVC, LakeFS, object storage with versioning |
| Randomness | seeded RNGs, deterministic algorithms |

### 2.2 Environment management

- **Don't** rely on system Python.
- **Do** pin versions: `pip freeze > requirements.txt` is a starting point; for stricter pinning use `pip-tools` or `uv`.
- Containerize for "works the same anywhere" — see [Day 2, Module 5](../day2/day2_materials.md#module-5--packaging--serving-models).

### 2.3 Project structure

A common, opinionated layout (e.g. Cookiecutter Data Science):

```
project/
├── data/            # raw, interim, processed (gitignored)
├── notebooks/       # exploration only
├── src/             # importable Python package
│   ├── data/
│   ├── features/          # feature engineering code (transforms, pipelines, encoders)
│   ├── models/
│   └── inference/
├── tests/
├── pyproject.toml
└── README.md
```

> **Best practice:** notebooks for exploration; production code lives in `src/` and is unit-tested.

### 2.4 Seeds and determinism

```python
import os, random, numpy as np
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
# torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
```

Even then full determinism is hard (GPU non-determinism, parallel data loaders). Document what you can guarantee.

➡ **Lab 1** in [day1_practice.ipynb](day1_practice.ipynb#Lab-1) — make a baseline notebook reproducible.

---

## Module 3 — Experiment Tracking with MLflow (13:15 – 14:45)

### 3.1 The problem

Without tracking, you end up with:

- `model_final.pkl`, `model_final_v2.pkl`, `model_final_v2_REAL.pkl`
- "What learning rate gave us 0.91 AUC last Tuesday?"
- Two team members reporting different numbers for the "same" model.

### 3.2 What an experiment tracker stores

For each **run**:

- **Parameters** (hyperparameters, dataset version, code commit).
- **Metrics** (loss, accuracy, AUC, latency...).
- **Artifacts** (model file, plots, confusion matrix).
- **Tags / metadata** (author, git SHA, environment).

### 3.3 MLflow at a glance

Four components:

1. **Tracking** — log params/metrics/artifacts.
2. **Projects** — package code in a reproducible form.
3. **Models** — a standard format for packaging models.
4. **Model Registry** — versioned, stage-aware (`Staging`, `Production`).

### 3.4 Minimal API

```python
import mlflow

mlflow.set_experiment("churn-baseline")
with mlflow.start_run():
    mlflow.log_param("C", 1.0)
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_metric("auc", 0.87)
    mlflow.log_metric("accuracy", 0.83)
    mlflow.sklearn.log_model(model, "model")
```

### 3.5 Tracking → Registry workflow

```
     log runs ──▶ compare runs ──▶ pick best ──▶ register model ──▶ stage: Staging ──▶ Production
```

➡ **Lab 2** in [day1_practice.ipynb](day1_practice.ipynb#Lab-2) — train multiple models and compare them in MLflow.

---

## Module 4 — Data: Versioning, Validation, Pipelines (15:00 – 16:30)

### 4.1 Why version data?

Models are functions of **data + code + config**. If data changes silently, results are not reproducible. Examples of silent change:

- Upstream team renames a column.
- A new country is added to the source.
- Label definition changes ("active user" now means 30 days, not 7).

### 4.2 Approaches

| Approach | Tool | Trade-offs |
|----------|------|------------|
| Snapshots in object storage | S3 versioning | Simple, but no metadata |
| Git-like data versioning | **DVC**, LakeFS | Diffs, branches, ties to git |
| Data warehouse time travel | Snowflake, BigQuery, Delta Lake | Powerful, vendor-bound |
| Feature stores | Feast, Tecton | Reuse + consistency train/serve |

### 4.3 Data validation

Before training **and** before serving, validate:

- Schema (columns, types).
- Distribution (means, ranges, null rates).
- Referential integrity.

Tools: **Great Expectations**, **Pandera**, **TFDV**, **Evidently**.

### 4.4 Train/serve skew

A leading cause of production failures: features computed differently in training vs serving. Mitigations:

- Shared feature code (one library used by both).
- Feature store.
- Logging serving features and replaying them in training.

### 4.5 Pipelines

A **pipeline** is a DAG of steps: ingest → validate → feature → train → evaluate → register. Tools: Airflow, Prefect, Dagster, Kubeflow Pipelines, Metaflow, ZenML.

Even without a fancy tool, **start with a single `make train` or `python -m src.pipeline.train`** that does everything end-to-end. If you can't run your training with one command, you don't have a pipeline.

➡ **Lab 3** in [day1_practice.ipynb](day1_practice.ipynb#Lab-3) — write a tiny end-to-end pipeline as Python functions and add basic data validation.

---

## Industry Best Practices — Day 1 Summary

1. **Treat ML projects like software projects**: version control, code review, tests, CI.
2. **One command to reproduce**: `make train` or equivalent. No "first run cell 3, then cell 7".
3. **Log everything that affects the result**: code commit, data version, params, metrics, environment.
4. **Validate data at boundaries** — never trust upstream silently.
5. **Notebooks for exploration, modules for production.**
6. **Start small.** A README + `requirements.txt` + MLflow + a single training script beats an unfinished Kubeflow setup.

---

## Discussion Questions

1. Of the four pillars of reproducibility, which is weakest in your current workflow?
2. Pick a model you've built. What would it take to retrain it tomorrow on fresh data — in minutes, not days?
3. What's the simplest data validation that would have caught your last "bad model" incident?

## Further Reading

- D. Sculley et al., *Hidden Technical Debt in Machine Learning Systems*, NeurIPS 2015.
- Google Cloud, *MLOps: Continuous delivery and automation pipelines in machine learning*.
- C. Huyen, *Designing Machine Learning Systems*, O'Reilly 2022.
