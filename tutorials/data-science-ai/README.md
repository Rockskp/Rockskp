# Data Science & AI — Bilingual (English + हिन्दी) Study Guide

One combined guide: **[STUDY-GUIDE.md](STUDY-GUIDE.md)** (all modules in one file), plus one file per module and executed notebooks in `notebooks/`.

| Module | Status |
|---|---|
| 1 AI environment · 2 Python · 3 Math/stats · 4 NumPy/pandas/SQL/viz · 5 Feature engineering | Full lessons, code verified |
| 6 Supervised ML · 7 Trees/ensembles/SHAP · 8 Unsupervised/anomalies/recommenders | Full lessons, code verified |
| 9–18 + 8 capstones | Compact outline ([module-09-18-roadmap.md](module-09-18-roadmap.md)) — to be expanded |

Every full lesson has: concept in English and Hindi, an analogy, the math with a worked example on real data, runnable code, interview traps, memory aids (mnemonic, "याद रखो" rule, 5 flashcards), 3 exercises with hidden solutions and a 5-question quiz. Every module ends with a cheat sheet, a mini-project and a spaced-repetition plan (Day 1, 3, 7, 21, 60).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r tutorials/data-science-ai/requirements.txt
cd tutorials/data-science-ai
python tools/build_notebooks.py          # rebuilds and executes every notebook (the verification step)
```

No paid API or GPU needed. Verified with Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9, PyTorch 2.15 (CPU).
Datasets in `datasets/` come from seaborn-data (BSD-3). Course syllabus was used for scope and order only; content is original, with sources cited per module.
