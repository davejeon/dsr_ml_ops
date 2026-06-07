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

| Pillar | Definition | Tool examples |
|--------|------------|---------------|
| **Code** | Every change to source code is tracked so you can pinpoint exactly which version produced a given result, review changes before merging, and roll back if needed. | `git`, code review (PRs/MRs) |
| **Environment** | The exact set of software dependencies (Python version, library versions, system libraries) is captured and can be recreated identically on any machine. | `venv`, `conda`, `pip-tools`, Docker |
| **Data** | The dataset used for training is versioned and addressable, so you can always re-run an experiment against the same snapshot of data that produced the original result. | DVC, LakeFS, S3 versioning |
| **Randomness** | All sources of stochasticity (weight initialization, data shuffling, dropout, train/val splits) are controlled via fixed seeds so runs are deterministic given the same inputs. | seeded RNGs, deterministic algorithms |

> **Why all four matter:** fixing only code but not data means you still can't reproduce yesterday's run if the upstream table was updated. Fixing all four means a colleague — or future you — can clone the repo, run one command, and get the same numbers.

### 2.2 Environment management

- **Don't** rely on system Python.
- **Do** pin versions: `pip freeze > requirements.txt` is a starting point; for stricter pinning use `pip-tools` or `uv`.
- Containerize for "works the same anywhere" — see [Day 2, Module 5](../day2/day2_materials.md#module-5--packaging--serving-models).

#### Tool comparison: venv vs conda vs pip-tools vs Poetry vs Docker

| Tool | What it isolates | Manages Python itself? | Pin mechanism | Best for |
|------|-----------------|----------------------|---------------|----------|
| `venv` | Python packages only | No — uses whichever `python` you call it with | `requirements.txt` (manual or `pip freeze`) | Simple projects; already have the right Python version |
| `conda` | Packages **and** Python version **and** C/system libraries | Yes — can install Python 3.9, 3.11, etc. per env | `environment.yml` | Data science; packages with complex C/CUDA deps (e.g. `numpy`, `torch`) |
| `pip-tools` | Packages only (no Python, no system libs) | No | `requirements.in` → compiled `requirements.txt` with full hash-locked tree | Teams who need reproducible, auditable pip installs without conda overhead |
| `poetry` | Packages only (no Python, no system libs) | No (but integrates with `pyenv`) | `pyproject.toml` → `poetry.lock` (hashed) | Modern pure-Python projects; all-in-one dep management, venv, and packaging |
| Docker | Everything — OS, system libs, Python, packages | Yes — the whole OS is frozen | `Dockerfile` + `requirements.txt` inside | Deployment; CI; sharing with people who aren't Python users |

**Mental model:**

```
venv        — isolates pip packages from your system
conda       — isolates pip packages + Python version + system libs
pip-tools   — generates a fully-pinned, hash-verified pip lockfile
poetry      — all-in-one: manages the venv + deps + lockfile + packaging
Docker      — freezes the entire OS + runtime; nothing leaks in or out
```

**What is a pip lockfile?**

A **lockfile** is a machine-generated file that records the *exact* version and cryptographic hash of every package (direct and transitive) that was installed at a known-good point in time.

- Your `requirements.in` lists what you *care about* (e.g. `scikit-learn>=1.3`).
- `pip-compile` resolves the full dependency tree — including scikit-learn's own deps (`numpy`, `scipy`, `joblib`, etc.) — and writes a `requirements.txt` that pins every single package to an exact version plus a SHA-256 hash:

```
scikit-learn==1.4.2 \
    --hash=sha256:a3b4c... \
    --hash=sha256:d5e6f...
numpy==1.26.4 \
    --hash=sha256:7a8b9...
```

- `pip-sync` installs *exactly* that set — nothing more, nothing less.

The hash check means pip refuses to install a package if its contents don't match, protecting against supply-chain attacks (a malicious package swapped in at the same version number). This is stricter than a plain `pip freeze`, which pins versions but does not verify hashes.

**Poetry vs pip-tools:** Both produce a hashed lockfile, but they differ in scope:

- **pip-tools** is just the compile/sync step — you manage the venv yourself and the format stays `requirements.txt`, which is universally understood.
- **Poetry** is all-in-one: it creates and manages the venv, resolves deps, writes `poetry.lock`, *and* can build/publish packages (`poetry build`, `poetry publish`). The trade-off is a steeper learning curve and a `pyproject.toml`-only workflow that some CI tools handle less gracefully.

For ML projects, pip-tools is often preferred because the conda/Docker stack already handles the venv layer, so Poetry's extra tooling adds complexity without benefit. Poetry shines for pure-Python libraries or services where you also need to publish to PyPI.

**Choosing in practice:**
- Start with `venv` + `pip-tools` for pure-Python projects where simplicity matters.
- Use `poetry` when building a pure-Python library or service and you want one tool to handle deps, venv, and packaging.
- Reach for `conda` when you need a specific Python version or heavy C/CUDA dependencies.
- Add Docker when you need to hand off to an environment you don't control (CI server, colleague's machine, production container).
- `conda` + Docker is common in ML: conda resolves tricky deps, Docker guarantees the image is identical everywhere.

```bash
# venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# conda
conda env create -f environment.yml
conda activate my-env

# pip-tools: compile then install
pip-compile requirements.in        # writes requirements.txt with hashes
pip-sync requirements.txt          # installs exactly that, nothing more

# poetry: install deps (creates venv automatically)
poetry install                     # reads pyproject.toml, writes/respects poetry.lock
poetry add scikit-learn            # adds dep, updates lock
poetry run python -m src.train     # run inside managed venv

# Docker
docker build -t my-ml-project .
docker run --rm my-ml-project python -m src.pipeline.train
```

### 2.3 Project structure

A common layout (e.g. Cookiecutter Data Science):

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

#### Snapshots in object storage

The simplest approach: every time you prepare a dataset, write it to a new path or use a storage system that keeps old versions automatically.

- **S3 versioning** — enabling versioning on an S3 bucket means every `PUT` keeps the previous object. You can retrieve any past version by specifying its version ID. No extra tooling required.
- **Naming conventions** — many teams simply timestamp the path: `s3://my-bucket/data/churn/2024-06-07/train.parquet`. Cheap, auditable, but you have to manage retention manually.
- **Limitations** — there's no metadata about *why* a version was created, no diffing, and no concept of branches. When you have dozens of experiments pointing at different snapshots, tracking which run used which file becomes its own problem.

#### Git-like data versioning

These tools bring the git mental model (commits, branches, diffs, remote remotes) to large files and datasets.

**DVC (Data Version Control)**
- Works alongside git: DVC stores a small `.dvc` pointer file in git while the actual data lives in a remote (S3, GCS, Azure, local).
- `dvc add data/train.csv` → creates `data/train.csv.dvc` (committed to git) + pushes bytes to the remote.
- `dvc repro` re-runs the pipeline if inputs change; `dvc push/pull` sync data the same way `git push/pull` syncs code.
- The result: a single git commit captures code *and* the pointer to the exact data snapshot used.

**Installation:** DVC is a Python package and must be explicitly installed — without it, `.dvc` pointer files are just inert YAML. Git will track them, but nothing will know how to fetch the actual data they point to. The `dvc` CLI is what reads those files and performs `dvc pull` (download data), `dvc push` (upload data), `dvc checkout` (swap files after a `git checkout`), and `dvc repro` (re-run the pipeline).

```bash
pip install dvc          # core only (local remote)
pip install "dvc[s3]"    # + S3 support
pip install "dvc[gs]"    # + GCS support
pip install "dvc[all]"   # all remote backends
```

In practice this means:
- Every developer and every CI machine needs `dvc` installed alongside the project's normal Python deps — so it typically goes in `requirements.txt` or `pyproject.toml`.
- The remote (S3, GCS, etc.) also needs credentials configured, either via the usual cloud SDK env vars (`AWS_ACCESS_KEY_ID`, etc.) or `dvc remote modify`. DVC intentionally does not store credentials in the repo.
- There is no IDE plugin required; it is purely a CLI tool.

```bash
dvc init
# Creates the .dvc/ directory (config, cache) and auto-updates .gitignore.
# Always commit .dvc/config so teammates know which remote to use.

dvc remote add -d myremote s3://my-bucket/dvc-store
# -d marks this as the *default* remote. You can have multiple remotes
# (e.g. a fast local cache + a shared S3 remote) and target specific ones
# with --remote <name> on push/pull.

dvc add data/raw/churn.csv
# Two things happen:
#   1. The file is copied into .dvc/cache/ (content-addressed by MD5 hash).
#   2. data/raw/churn.csv.dvc is written — a tiny YAML with the hash and size.
# The real CSV is also appended to .gitignore so it can never be committed to git.
# Caveat: if you edit the file directly without re-running dvc add, the pointer
# goes stale and teammates will pull the old version. Treat dvc add like git add.

git add data/raw/churn.csv.dvc .dvc/config
git commit -m "track raw churn dataset v1"
# The pointer file is < 1 KB and safe to commit.
# Key invariant: git tracks *what* the data is (its hash); DVC tracks *where* the bytes live.

dvc push
# Uploads only files whose hashes aren't already in the remote — DVC is content-addressed,
# so uploading the same bytes twice is a no-op.
# Caveat: dvc push does NOT run automatically after git commit. You must call it explicitly
# or wire it into a CI step, otherwise teammates can git pull the pointer but not the data.
```

**Caveats and gotchas:**

- **`.gitignore` is mutated silently** — `dvc add` appends the tracked file to `.gitignore`. This can look like git is ignoring a file for no reason if you aren't expecting it.
- **Large file rewrites are expensive** — changing even one byte of a 10 GB file creates a completely new cache entry. Plan versioning granularity to match your actual change cadence.
- **`dvc pull` is separate from `git pull`** — a new team member who only runs `git clone` + `git pull` will have the pointer files but no data. They must also run `dvc pull`. Wrapping both in a `make setup` target avoids confusion.
- **`dvc add` vs `dvc.yaml` pipelines** — the example above tracks a single file. For a full pipeline (ingest → clean → train), define stages in `dvc.yaml` and run `dvc repro`, which caches intermediate outputs and only re-runs stages whose inputs changed. More powerful, but more overhead to set up.

**LakeFS**
- Acts as a git-like layer *on top of* existing object storage (S3, GCS, Azure).
- You get branches, commits, and merges for your data lake — without copying bytes.
- Teams use feature branches for data experiments and merge them when validated, exactly like code review for data.
- Heavier infrastructure than DVC; better suited to large organisations with a dedicated data platform team.

#### Data warehouse time travel

**What is a data warehouse?**

A **data warehouse** is a centralised repository that stores large volumes of structured, historical data from multiple source systems (CRMs, transaction databases, event logs, etc.), optimised for analytical queries rather than transactional writes. Data is typically loaded in batches via ETL/ELT pipelines and organised into wide, denormalised tables so analysts and ML pipelines can query large date ranges quickly. Examples: Snowflake, BigQuery, Amazon Redshift, Azure Synapse.

**What is a data mart?**

A **data mart** is a focused subset of a data warehouse scoped to a specific business domain or team — e.g. a "Marketing data mart" or a "Churn prediction data mart". Where a warehouse holds everything, a mart holds only the tables and columns relevant to one use case, often with pre-aggregated views to speed up common queries. Data marts are typically read by a single team and are built on top of the warehouse rather than replacing it.

```
Raw sources → Data warehouse (all domains) → Data mart (one domain) → ML feature table
```

For ML work you'll usually pull training data from a mart or a curated feature table in the warehouse, not directly from raw source tables.

**Time travel** is a feature of modern analytical databases that keeps a change history of every table, letting you query data *as it was at a point in time*. This matters for reproducibility: if your training data changes between runs, you need to be able to go back to the exact snapshot the original model saw.

- **Snowflake** — `SELECT * FROM my_table AT(TIMESTAMP => '2024-06-01 00:00:00')` retrieves the table as it looked at that moment. Retention window is configurable (up to 90 days on Enterprise). Also supports `AT(OFFSET => -3600)` (seconds ago) and `AT(STATEMENT => '<query_id>')`.
- **BigQuery** — `FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)` queries; also supports scheduled snapshots as separate tables. Retention is 7 days by default.
- **Delta Lake / Apache Iceberg** — open formats that add ACID transactions and time travel to Parquet files on object storage. `spark.read.format("delta").option("versionAsOf", 42).load(path)` reads version 42 of the table. Unlike the managed platforms above, these work on your own object storage with no vendor lock-in.

The advantage is zero extra tooling if you're already on these platforms. The disadvantages are:
- Your ML code becomes tightly coupled to the vendor's query syntax.
- You can't reproduce a run if the retention window has expired (Snowflake's default is only 1 day on the Standard tier).
- Time travel is not a substitute for explicitly recording *which* snapshot a training run used — you still need to log the timestamp or version number as an MLflow parameter.

#### Feature stores

A feature store is a centralised repository that computes, stores, and serves features — the transformed columns your model actually trains on.

**The core problem: train/serve skew**

When you train a model, you compute features in a batch pipeline — usually in Python or SQL, running overnight. When the model serves predictions in production, the same features need to be recomputed in real time, often in a different codebase or language. If there is *any* difference in that logic (different null handling, different rounding, different time window), the model receives inputs it has never seen during training. It silently degrades because the distribution of inputs has shifted — not because the data changed, but because the *transformation* changed. This is **train/serve skew**, and it is one of the leading causes of models that look fine in evaluation but underperform in production.

**Why they matter for versioning:** without a feature store, you often compute the same feature differently in the training pipeline (batch, offline) and the serving layer (real-time, online). This mismatch is called **train/serve skew** and is a leading cause of silent model degradation in production.

A feature store solves this with:

- **One definition, two paths** — you write the feature logic once (e.g. "30-day rolling average spend per customer"). The store internally maintains:
  - an **offline store** (e.g. a warehouse table or Parquet files on S3) used by training pipelines to fetch large historical windows of feature values
  - an **online store** (e.g. Redis, DynamoDB, Bigtable) used by the serving layer to look up the latest value for a single entity in milliseconds

  Both are populated by the *same* compute job, so the transformation is guaranteed identical across training and serving.

- **Point-in-time correct joins** — when building a training dataset, you join a label table (e.g. "did customer churn on day X?") with feature values. The naive join uses the most recent feature value — but that may include information from *after* the label date, leaking future data into training. A feature store performs the join at the label's timestamp, retrieving only feature values that were known at that exact moment. Without this, your offline evaluation metrics are inflated and the model underperforms in production.

- **Versioning** — feature definitions are versioned; you can retrain against any past version.

| Tool | Type | Notes |
|------|------|-------|
| **Feast** | Open source | Self-hosted; you bring your own offline store (BigQuery, Redshift, files) and online store (Redis, SQLite). Low cost to start, but you own the infrastructure. |
| **Tecton** | Managed SaaS | Fully managed; strong on streaming / real-time features from Kafka or Kinesis. Enterprise pricing. |
| **Vertex AI Feature Store** | Managed (GCP) | Tight BigQuery integration; easiest if you're already on GCP. |
| **AWS SageMaker Feature Store** | Managed (AWS) | Tight S3/Redshift integration; easiest if you're already on AWS. |
| **Databricks Feature Store** | Managed | Delta Lake native; integrates with MLflow for automatic feature lineage tracking. |

**When you actually need one:** most small projects don't. The tipping point is usually one of:
1. Multiple models consuming the same feature — without a store, the same logic is duplicated and diverges over time.
2. Sub-100ms serving latency requirements — you can't recompute complex features on the fly.
3. Compliance requirements to explain exactly what data a model used at prediction time.

Feature stores are overkill for most small teams but become essential when multiple models share features or when online latency is critical.

### 4.3 Data validation

Before training **and** before serving, validate:

- **Schema** — are the expected columns present? Do they have the right types? Are required fields non-null?
- **Distribution** — have the statistical properties shifted? Is `age` now capped at 99 when it used to go to 120? Has a previously rare category suddenly become the majority class?
- **Referential integrity** — do foreign keys resolve? Are join keys consistent across tables?

Catching these *before* training avoids wasting compute on a corrupted dataset. Catching them *before serving* avoids making predictions on inputs the model has never seen.

#### Tool comparison

**Pandera** *(MIT, open source)*
- Validates pandas and Polars DataFrames by decorating them with a typed schema.
- Best for: **in-pipeline validation** — drop it in at the top of a training script to assert column types, ranges, and null rates before any processing begins.
- Trade-offs: lightweight and Pythonic, but schema definitions live in code rather than a shareable test suite. Less suited to generating reports for non-technical stakeholders.

**Great Expectations (GX)** *(Apache 2.0, open source; GX Cloud is paid)*
- Defines a suite of "expectations" (assertions) about your data and generates HTML validation reports.
- Best for: **CI/CD data quality gates** — block a pipeline from proceeding if the incoming data fails expectations. Good for teams that need audit trails and shareable reports.
- Trade-offs: significant setup overhead (data sources, checkpoints, data docs). Overkill for a single-developer project. The managed GX Cloud tier removes infrastructure burden at a cost.

**TFDV (TensorFlow Data Validation)** *(Apache 2.0, open source)*
- Computes statistics over a dataset and compares them against a saved schema or a reference dataset to detect drift.
- Best for: **TFX / TensorFlow pipelines** where you need tight integration with the rest of the TFX ecosystem.
- Trade-offs: tightly coupled to TensorFlow and TFX; awkward to use outside that stack. If you're not already on TFX, the dependency cost is not worth it.

**Evidently** *(Apache 2.0, open source; Evidently Cloud is paid)*
- Generates interactive reports comparing a reference dataset (e.g. training data) to a current dataset (e.g. last week's production traffic) across data drift, model performance, and data quality dimensions.
- Best for: **production monitoring** — run it on a schedule to detect when the data your model is seeing in production has drifted from what it trained on.
- Trade-offs: designed for monitoring, not pre-training validation. It won't block a pipeline step; it produces a report for a human to review. The open-source library requires you to build your own alerting layer on top.

#### Choosing in practice

| Situation | Recommended tool |
|-----------|------------------|
| Validate DataFrame schema at start of training script | Pandera |
| CI gate to block training if upstream data is bad | Great Expectations |
| TFX / TensorFlow pipeline | TFDV |
| Monitor production traffic for drift vs training baseline | Evidently |
| All of the above on a large team | GE for CI gates + Evidently for monitoring |

These tools are complementary, not mutually exclusive. A mature pipeline might use Pandera for lightweight in-code checks, GE for the CI gate, and Evidently for weekly drift reports.

### 4.4 Train/serve skew

A leading cause of production failures: features computed differently in training vs serving. The model was trained on one version of a feature and then asked to predict on a slightly different version — same column name, different logic. The error is silent: no exception is raised, predictions just quietly get worse.

**Mitigations, from simplest to most robust:**

- **Shared feature code** — extract all feature transformations into a single importable Python module (e.g. `src/churn/features/build.py`) and import it in both the training pipeline and the serving layer. If the logic changes, it changes in one place. This is the minimum viable mitigation and is appropriate for most small-to-medium projects.

- **Shared preprocessing artifact** — fit your sklearn `Pipeline` or `ColumnTransformer` on training data and serialise it (e.g. `joblib.dump(preprocessor, 'preprocessor.pkl')`). At serving time, load and apply the *same fitted object*. This guarantees identical imputation values, scaling parameters, and encoding mappings — not just identical code.

- **Feature store** — as covered in 4.2, a feature store enforces one definition and two paths (offline for training, online for serving) at infrastructure level. The strongest guarantee, but the highest operational cost.

- **Logging serving features and replaying them in training** — log the raw feature vector that was passed to the model at prediction time (not the raw input, the *post-transformation* vector). Periodically retrain on these logged vectors rather than re-deriving features from raw data. This closes the skew loop completely but adds storage and pipeline complexity.

### 4.5 Pipelines

A **pipeline** is a DAG of steps: ingest → validate → feature → train → evaluate → register. Tools: Airflow, Prefect, Dagster, Kubeflow Pipelines, Metaflow, ZenML.

Even without a fancy tool, **start with a single `make train` or `python -m src.pipeline.train`** that does everything end-to-end. If you can't run your training with one command, you don't have a pipeline.

#### What is a DAG?

A **Directed Acyclic Graph (DAG)** is a set of tasks with dependencies between them that form no cycles. "Directed" means each edge has a direction (A runs before B); "Acyclic" means you can never follow the edges and loop back to where you started (no circular deps).

In pipeline tools, you define the graph structure in code — the *what* and *in what order* — and the tool's scheduler handles the *when*, *where*, and *retry logic*. This separation is what makes orchestration tools valuable: your code describes business logic, the tool handles operations.

```
load_data ──▶ validate ──▶ featurize ──▶ train_model ──▶ evaluate
                                              │
                                              ▼
                                        register_model
```

#### Apache Airflow

Airflow is the most widely deployed open-source pipeline orchestrator in data and ML engineering. It was created at Airbnb in 2014, donated to the Apache Software Foundation in 2016, and is now the de-facto standard in data platforms.

**Core concepts:**

| Concept | What it is |
|---------|-----------|
| **DAG** | A Python file that defines a workflow as a graph of tasks. Each DAG has a unique `dag_id`, a `start_date`, and a `schedule`. |
| **Task / Operator** | A single unit of work. Built-in operators cover `PythonOperator`, `BashOperator`, `SQLExecuteQueryOperator`, HTTP calls, Spark, dbt, and dozens more. |
| **TaskFlow API** | Modern (Airflow 2.0+) way to write DAGs using `@dag` and `@task` decorators instead of verbose operator instantiation. Cleaner and more Pythonic. |
| **XCom** | "Cross-communication" — the mechanism tasks use to pass small values to each other. A task returns a value; a downstream task receives it as a function argument. **Only use XComs for small values** (metadata, paths, IDs); pass large data through files or object storage. |
| **Scheduler** | A daemon that reads DAG files, determines which task instances are due, and submits them to the executor. |
| **Executor** | How tasks are actually run: `LocalExecutor` (parallel subprocesses on one machine), `CeleryExecutor` (distributed workers via Redis/RabbitMQ), `KubernetesExecutor` (one pod per task). |
| **Connection** | Credentials for external systems (databases, cloud providers, APIs) stored encrypted in the Airflow metadata DB. Referenced by `conn_id` in operators — keeps secrets out of DAG code. |
| **Variable** | Key-value store in the Airflow metadata DB for runtime configuration (e.g. `batch_size`, `s3_bucket`). |
| **Sensor** | A special operator that polls for a condition (file exists, S3 key appears, upstream job finishes) before allowing downstream tasks to run. |

**Airflow architecture:**

```
 ┌───────────────┐          ┌──────────────────────────────────┐
 │  DAG files    │──parse──▶│          Scheduler               │
 │  (Python)     │          │  (reads DAGs, creates task inst.) │
 └───────────────┘          └────────────┬─────────────────────┘
                                         │ submit
                            ┌────────────▼────────────┐
                            │        Executor          │
                            │  (LocalExecutor /        │
                            │   CeleryExecutor / K8s)  │
                            └────────────┬────────────┘
                                         │ run
                            ┌────────────▼────────────┐
                            │        Worker(s)         │
                            │  (execute task code)     │
                            └─────────────────────────┘
                                         │ metadata
                            ┌────────────▼────────────┐
                            │      Metadata DB         │
                            │  (SQLite / Postgres)     │
                            └─────────────────────────┘
```

The **Webserver** (UI) reads from the same metadata DB and provides the dashboard where you monitor runs, inspect logs, trigger DAGs manually, and manage connections.

**A minimal Airflow DAG (TaskFlow API):**

```python
from __future__ import annotations
from datetime import datetime
import pandas as pd
from airflow.decorators import dag, task

@dag(
    dag_id="iris_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="0 6 * * *",   # daily at 06:00
    catchup=False,
)
def iris_pipeline():

    @task
    def load_data() -> str:
        df = pd.read_csv("/data/iris.csv")
        return df.to_json(orient="records")   # XCom: returned value stored in DB

    @task
    def validate(raw_json: str) -> str:       # XCom: received as argument
        df = pd.read_json(io.StringIO(raw_json), orient="records")
        assert df.isnull().sum().sum() == 0
        return raw_json

    @task
    def train(raw_json: str) -> float:
        df = pd.read_json(io.StringIO(raw_json), orient="records")
        # ... train model ...
        return accuracy

    raw = load_data()          # each @task call returns an XComArg
    clean = validate(raw)      # dependency: validate runs after load_data
    train(clean)               # dependency: train runs after validate

iris_pipeline()   # instantiate the DAG
```

**Key behaviours to know:**

- **`schedule`** accepts cron strings (`"0 6 * * *"`), `timedelta` objects, or `None` (manual trigger only). `catchup=False` means Airflow won't try to backfill all past intervals since `start_date`.
- **`dags test`** runs a single DAG run synchronously in the current process — no scheduler, no database required. This is the fastest way to test logic locally.
- **Tasks are isolated** — each task runs in its own process (or container). Never assume shared in-memory state between tasks; use XComs or files.
- **Idempotency** — tasks should be safe to re-run. If a run fails halfway through and you retry, already-succeeded tasks are skipped; only the failed task (and its dependents) re-run.

#### Airflow vs other orchestrators

| Tool | Strengths | Weaknesses | Best for |
|------|-----------|-----------|----------|
| **Airflow** | Mature, huge community, rich operator ecosystem, powerful UI | Steep learning curve, verbose (pre-2.0), scheduler overhead | Data platforms, ML pipelines at scale |
| **Prefect** | Python-native, minimal boilerplate, good local dev UX | Smaller ecosystem than Airflow | Teams wanting quick iteration with a modern API |
| **Dagster** | Strong asset-based model, built-in data cataloguing | Biggest conceptual shift from plain Python | Teams that want traceability and data assets as first-class citizens |
| **ZenML** | ML-focused, step caching, stack abstraction | Less general-purpose | Pure ML pipelines where you want reproducibility without infra setup |
| **Metaflow** | Data scientist-friendly, notebook-first | Tied to AWS; UI requires Metaflow Service | Netflix-style large-scale ML on AWS |
| **Make / plain Python** | Zero dependencies, easy to understand | No retries, no scheduling, no parallelism | Single-developer projects; early PoC |

**Rule of thumb:** start with a `Makefile` or a plain Python script. Graduate to an orchestrator when you need scheduling, retries, parallelism, or observability across multiple pipelines.

➡ **Session 5** in [Session_5/airflow_pipeline.ipynb](Session_5/airflow_pipeline.ipynb) — write and run a complete Airflow DAG for the iris dataset end to end.

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
