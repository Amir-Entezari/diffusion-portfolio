# Diffusion Portfolio

Research implementations of conditional joint-return forecasting and portfolio
evaluation. Current components include Gaussian and alpha-stable diffusion,
MLP and Neural CDE history encoders, evidential regime probes and routing,
topology and covariance/SPD feature ablations, and the Diffolio literature baseline.

The package is organized by function: `data/` owns return panels, sources,
normalization and windows; `models/` owns encoders, diffusion, uncertainty,
topology and geometry; `baselines/diffolio/` owns the baseline and its adapters;
`training/`, `evaluation/` and `portfolio/` own optimization and diagnostics.
Scripts parse arguments and call these modules. Proposal stages are research
hypotheses, not software-package phases; no unimplemented proposal components
are included.

## Install and test

```bash
pip install -e '.[dev,research]'
pytest -q
python -m compileall src scripts
```

For MLP-only use, `pip install -e .` is sufficient. CDE and topology use the
existing optional `torchcde` and `gudhi` dependencies.

## Current experiments

```bash
python scripts/train_diffusion.py --config configs/cde.yaml --output-dir outputs/cde --resume
python scripts/evaluate_diffusion.py --run-dir outputs/cde

python scripts/extract_features.py --kind spd --cde-run-dir outputs/cde --output-dir outputs/spd_features
python scripts/train_feature_ablation.py --variant covariance --cde-run-dir outputs/cde --feature-dir outputs/spd_features --output-dir outputs/covariance --resume
python scripts/evaluate_diffusion.py --variant covariance --run-dir outputs/covariance --cde-run-dir outputs/cde --feature-dir outputs/spd_features

python scripts/train_diffusion.py --alpha 1.8 --config configs/levy.yaml --reference-run-dir outputs/cde --output-dir outputs/levy --resume
python scripts/evaluate_diffusion.py --run-dir outputs/levy
```

`configs/mvp.yaml` selects the MLP baseline. Feature extraction also supports
`--kind topology`, with `geometry` and `tda` ablations. Regime probes and routing
use `train_evidential_regime.py`, `train_router.py`, `configs/evidential.yaml` and
`configs/router.yaml`. Diffolio uses `train_diffolio.py` and `evaluate_diffolio.py`;
its configured Goyal workbook must exist at `data/raw/Data2025.xlsx`.

Forecast evaluation defaults to validation. Gaussian test evaluation is an
explicit `--split test` option; it is not part of refactor validation. Dates,
train-only normalization, objectives and checkpoint keys retain their original
semantics. Existing diffusion training always uses train Z-scores, even for
`return_scaling: none`; this historical quirk is preserved. Diffolio honors its
explicit scaling setting. Generic `ReturnTable` requires asset column names,
and generic model builders take the asset count from the data.

Notebooks under `cde/`, `topology/`, `spd/` and `baselines/` orchestrate existing
experiments. Historical results and Kaggle artifact paths retain their original
names. `scripts/kaggle_artifacts.py` restores and backs up that separate store.
State-dict checkpoints need no key migration; legacy CDE/conditioning and KF12
import shims only re-export canonical implementations.
