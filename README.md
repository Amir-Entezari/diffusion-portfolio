# Diffusion Portfolio Research MVP

Research code for testing whether diffusion-based joint-return modeling provides useful
information for portfolio construction.

This repository intentionally separates the **scientific MVP** from the much larger proposal.
The proposal components are hypotheses to be tested one by one, not assumed-valid phases.

## Current stage

See [`ROADMAP.md`](ROADMAP.md). The current task is repository refactoring before the
Kenneth French 12 Industry Portfolios data pipeline is added.

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

## Historical scaffold

The untouched AI-generated code received from Matin is preserved in git tag:

```bash
git checkout matin-scaffold-v0
```

Do not treat that tag as scientifically validated.
