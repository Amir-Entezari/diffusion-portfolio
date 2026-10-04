# Diffusion Portfolio Research MVP

Research code for testing whether diffusion-based joint-return modeling provides useful
information for portfolio construction.

This repository intentionally separates the **scientific MVP** from the much larger proposal.
The proposal components are hypotheses to be tested one by one, not assumed-valid phases.

## Current stage

Research components are implemented and validated incrementally through experiment
configs and scripts. Phase names describe experiment stages; reusable source code
remains organized by model/component rather than by phase.

## Install

```bash
pip install -e .
```

For CDE/TDA research components:

```bash
pip install -e '.[research]'
```

For tests:

```bash
pip install -e '.[dev]'
pytest
```

## Kaggle principle

The notebook is only an orchestrator. Source code lives in this repository. Kaggle should
clone/pull an exact commit, install the package, run tests, then execute scripts/configs.

Long-lived checkpoints and experiment outputs are stored separately from GitHub in one
private Kaggle Dataset. `scripts/kaggle_artifacts.py` is a small Kaggle-only helper for
restoring that artifact store once per session and backing up experiment directories.
It is operational tooling and is not used by the model or training code.
