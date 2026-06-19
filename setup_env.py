#!/usr/bin/env python3
"""Run this once to set up your virtual environment and pre-commit hooks."""

import platform
import subprocess
import sys
from pathlib import Path

VENV = Path(".venv")


def run(cmd):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def exe(name):
    """Resolve a venv executable cross-platform."""
    if platform.system() == "Windows":
        return VENV / "Scripts" / f"{name}.exe"
    return VENV / "bin" / name


if sys.version_info < (3, 10):
    sys.exit("Python 3.10+ is required.")

print("\n1. Creating virtual environment...")
if VENV.exists():
    print("   .venv already exists, skipping.")
else:
    run([sys.executable, "-m", "venv", ".venv"])

python = exe("python")

print("\n2. Installing dependencies...")
run([python, "-m", "pip", "install", "--upgrade", "pip", "-q"])
run([python, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

print("\n3. Registering Jupyter kernel...")
run([python, "-m", "ipykernel", "install", "--user", "--name", ".venv", "--display-name", ".venv"])

print("\n4. Pairing notebooks with Jupytext...")
notebooks = list(Path(".").rglob("*.ipynb"))
if notebooks:
    for nb in notebooks:
        run([python, "-m", "jupytext", "--set-formats", "ipynb,py:percent", str(nb)])
else:
    print("   No notebooks found, skipping.")

print("\n5. Installing pre-commit hooks...")
run([python, "-m", "pre_commit", "install"])

print("\nDone! Activate with:")
if platform.system() == "Windows":
    print("  .venv\\Scripts\\activate")
else:
    print("  source .venv/bin/activate")
