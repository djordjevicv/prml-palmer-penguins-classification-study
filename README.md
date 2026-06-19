# PRML: Comparative Study of kNN and Weighted kNN Classification on the Palmer Penguins Dataset

## Team — Group 6

| Index   | Name            |
|---------|-----------------|
| 66m/25  | Ivona Petrović  |
| 144m/25 | Vojin Đorđević  |

## Project Overview

Multi-class classification to predict penguin species (Adelie, Chinstrap, Gentoo) from four continuous morphological measurements: bill length, bill width, flipper length, and body mass. We compare standard majority-vote kNN against inverse-distance-weighted kNN, both implemented from scratch using NumPy only.

## Research Questions

* Does Weighted kNN outperform classic kNN on this dataset?
* What is the optimal k for each method?
* How much does feature standardization affect results?
* Where do the two methods disagree, and what does the confusion matrix reveal about the underlying class geometry?

## Repository Structure

```
prml-palmer-penguins/
├── data/                                  # raw dataset files (not committed)
├── figures/                               # plots exported from the notebook
├── notebooks/
│   ├── prml-project-group-6.ipynb        # main notebook — open this in Jupyter
│   └── prml-project-group-6.py           # Jupytext pair — do not edit directly
├── report/                                # final report document
├── src/                                   # reusable Python modules imported by the notebook
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── requirements.txt
└── setup_env.py
```

### notebooks/.ipynb and .py

`prml-project-group-6.ipynb` is the working file — the one you open and run in Jupyter.

`prml-project-group-6.py` is automatically generated and kept in sync by Jupytext on every commit. It stores only code and markdown — no outputs or metadata — making diffs clean and pull requests readable. Do not edit it directly.

## Getting Started

### Prerequisites

- **Python 3.10 or newer** — check with `python --version`. Download from [python.org](https://www.python.org/downloads/) if needed.
- **Git** — check with `git --version`. Download from [git-scm.com](https://git-scm.com/) if needed.

---

### Step 1 — Clone the repository

```
git clone https://github.com/djordjevicv/prml-palmer-penguins-classification-study.git
cd prml-palmer-penguins-classification-study
```

---

### Step 2 — Run the setup script

```
python setup_env.py
```

This creates the virtual environment, installs all dependencies, registers the Jupyter kernel, pairs the notebooks with Jupytext, and installs the pre-commit hooks.

---

### Step 3 — Activate the virtual environment

Do this every time you open a new terminal to work on the project.

**Windows (Command Prompt):**
```
.venv\Scripts\activate
```

**Windows (PowerShell):**
```
.venv\Scripts\Activate.ps1
```

> If PowerShell gives a permissions error, run this first:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**macOS / Linux:**
```
source .venv/bin/activate
```

Once activated you will see `(.venv)` at the start of your terminal prompt.

---

### Step 4 — Launch Jupyter

```
jupyter notebook
```

Open `notebooks/prml-project-group-6.ipynb` from the Jupyter file browser.

---

### Committing changes

The pre-commit hooks run automatically on every commit. They will:
1. Sync `prml-project-group-6.py` to match the current state of the notebook
2. Strip outputs and metadata from the `.ipynb` so it diffs cleanly

If a commit is rejected because the hooks modified files, stage the changes and commit again:

```
git add .
git commit -m "your message"
```