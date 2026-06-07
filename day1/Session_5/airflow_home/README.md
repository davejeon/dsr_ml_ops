# airflow_home

This directory is the `AIRFLOW_HOME` for the Session 5 Airflow demo.

**`airflow.cfg`, `airflow.db`, `logs/`, and `standalone_admin_password.txt` are
gitignored** because Airflow generates them with absolute paths at init time and
they are machine-specific.

## First-time setup

From the repo root, run:

```bash
export AIRFLOW_HOME="$(pwd)/day1/Session_5/airflow_home"
airflow db init
```

This regenerates `airflow.cfg` and `airflow.db` with paths correct for *your*
machine. The `dags/` directory (and `iris_pipeline.py` inside it) is committed
and uses `pathlib.Path(__file__)` — no absolute paths.

## Running the pipeline

```bash
export AIRFLOW_HOME="$(pwd)/day1/Session_5/airflow_home"

# Option 1: one-shot local test (no scheduler needed)
airflow dags test iris_pipeline 2024-01-01

# Option 2: full UI
airflow standalone
# open http://localhost:8080  (username: admin, password in standalone_admin_password.txt)
```
