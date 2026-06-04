# MLOps — 2 Day Course

A practical 2-day (≈ 8 h/day, with breaks) introduction to **Machine Learning Operations (MLOps)** for students with a data science / ML background and basic Python.

## Audience & Prerequisites

- Comfortable with Python basics (functions, classes, virtual environments).
- Familiar with at least one ML library (scikit-learn, PyTorch, or TensorFlow).
- Comfortable using a terminal and `git`.
- Laptop with Python 3.10+, `pip`, `git`, and Docker installed (Docker optional for some labs).

## Course Layout

| Day | Theme | Material | Practice |
|-----|-------|----------|----------|
| 1 | Foundations: lifecycle, reproducibility, tracking, data | [day1/day1_materials.md](day1/day1_materials.md) | [day1/day1_practice.ipynb](day1/day1_practice.ipynb) |
| 2 | Production: packaging, serving, CI/CD, monitoring, governance | [day2/day2_materials.md](day2/day2_materials.md) | [day2/day2_practice.ipynb](day2/day2_practice.ipynb) |

The markdown files contain **concepts, diagrams, and discussion questions**. The notebooks contain **hands-on labs** and are referenced from the markdown by section. The notebooks also link back to the relevant theory section.

## Suggested Daily Schedule (8 h with breaks)

| Time | Block |
|------|-------|
| 09:00 – 10:30 | Module A (lecture + lab) |
| 10:30 – 10:45 | Break |
| 10:45 – 12:15 | Module B (lecture + lab) |
| 12:15 – 13:15 | Lunch |
| 13:15 – 14:45 | Module C (lecture + lab) |
| 14:45 – 15:00 | Break |
| 15:00 – 16:30 | Module D (lecture + lab) |
| 16:30 – 17:00 | Recap, Q&A, homework |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Repository Layout

```
ml_ops/
├── README.md
├── requirements.txt
├── day1/
│   ├── day1_materials.md
│   └── day1_practice.ipynb
└── day2/
    ├── day2_materials.md
    └── day2_practice.ipynb
```
