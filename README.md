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

Return sources expose `ReturnTable(dates, returns, columns)` with arithmetic decimal returns
of shape `[time, assets]`; add a loader to the small `data/registry.py` dispatch
and select it in the data config. Encoders and diffusion builders receive the
asset count from that panel. Existing synthetic code is experimental: no generator
is selected, validated or integrated into this pipeline. A future adapter must
explicitly convert any log-return outputs to arithmetic decimal returns before
exposing a panel. Baseline-specific objectives and adapters belong together
under `baselines/`; reusable experiment functions belong in the package.

## Install and test

```bash
pip install -e '.[dev,research]'
pytest -q
python -m compileall src scripts
```

For MLP-only use, `pip install -e .` is sufficient. CDE and topology use the
existing optional `torchcde` and `gudhi` dependencies.

## French return datasets

The shared diffusion pipeline supports `ken_french_12`, `ken_french_49` and
`ken_french_100_size_bm`. The latter two load the official **daily value-weighted**
[49 Industry](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html)
and [100 Size × Book-to-Market](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_100_port_sz.html)
portfolios. Archives are cached under `data/raw/` and reused without refreshing.
Keep those archives alongside run artifacts: French can revise historical data.

`configs/cde_ff49.yaml` and `configs/cde_ff100.yaml` retain the CDE model, solver,
seeds, optimizer and calendar split settings from `configs/cde.yaml`.
`data.subtract_risk_free: true` subtracts exactly aligned daily French RF before
train-only normalization. It defaults to `true`, preserving KF12 behavior. A
future arithmetic-return source can set it to `false`; RF is then never loaded.
Source adapters must not return log returns or silently pre-subtract RF.

`data.source_options` forwards ordinary keyword arguments to the selected loader,
for example `path`, `download_if_missing: false`, or `missing_policy`. The new
French loaders reject missing sentinels by default. These two configs explicitly
choose `missing_policy: drop`: any date containing `-99.99` or `-999` is excluded
across **all** assets, with a warning reporting the count and affected date range.
No missing value is replaced by zero. Malformed rows, nonfinite values, unordered
dates and unexpected labels always fail. Exclusion occurs before sample slicing;
the actual available period can therefore start later than `sample_start`.

Shared diffusion windows use `data.max_gap_days: 7` by default. A larger gap
starts a new segment; each segment needs its own complete history and target.
Windows spanning a discontinuity are excluded with a warning, without filling
any returns. The check covers histories, the history/target join and multi-step
targets. Raw and standardized windows use the same dates. A segment too short
for a complete window contributes no samples; if none remain, preparation fails.
Normalization still fits all retained training observations only, and saved
scalers are reused exactly as before.

Seven days is a **coarse calendar-gap limit**, not an exchange calendar. It
permits ordinary weekends/holidays and the observed seven-day KF12 closure;
verified KF12 windows are unchanged. It cannot distinguish isolated missing
trading days from legitimate short closures. Choose a different positive limit
explicitly when another source's cadence requires it; increasing the limit to
bridge a data outage is not a continuity fix.

Availability in the cached official 202608 archives within the configured
1958–2023 sample, after seven-day continuity segmentation:

| Dataset | Retained observations | Excluded dates | Train / validation / test windows |
|---|---:|---:|---:|
| FF49 | 13,745 | 2,868 | 7,648 / 1,256 / 4,781 |
| FF100 Size × B/M | 13,197 | 3,416 | 8,550 / 68 / 4,159 |

FF49's first complete row is 1969-07-01. After that, exactly one source date is
missing through 2023: **1971-03-11**. Adjacent retained-date gaps have lengths
1/2/3/4/5/7 days, with counts 10,768/133/2,507/333/2/1; none exceed seven days.
FF49 is a practical full-training candidate under this declared coarse policy,
with 1,256 validation windows, but it is not a perfectly complete trading-day
panel. A stricter protocol could explicitly start after that missing date or
segment at known source-calendar omissions; neither change is applied here.

FF100 has **six** gaps exceeding a year: 366, 367, 367, 368, 1,670 and 1,828
calendar days. The last is from 1999-06-30 to 2004-07-01. Segmentation excludes
360 additional windows compared with the former row-adjacent construction.
Only **68 validation windows**, from 2004-09-27 to 2004-12-31, remain after a new
60-observation warmup. This is not an adequate basis for robust comparative
research across market conditions. Keep this config as a data/model smoke
reference until an explicit evaluation protocol is selected. Options are:

- retain all 100 portfolios but predeclare new split dates within sufficiently
  long continuous segments; this changes the benchmark and requires matched
  comparator periods;
- predeclare a smaller or different portfolio universe with better coverage;
  this is a different dataset, with selection-bias tradeoffs, not FF100;
- defer FF100 forecasting comparisons and use FF49 with its availability
  caveats. Do not use the existing held-out test period to tune a workaround.

Sample dates, split boundaries and portfolio universes are unchanged. Both
retained samples end on 2023-12-29; these configs are not matched-observation
KF12 comparisons. Old FF100 counts/results from windows crossing gaps are not
comparable to the corrected construction. Checkpoint/scaler formats are unchanged;
no historical artifact is rewritten. The smoke report shows usable window counts
(not an observation count inferred from windows), target ranges and target-date
gaps, which also include the post-discontinuity warmup. Smoke losses are not
forecasting results.

Initial checks (download once; no experiment training or forecast evaluation):

```bash
python scripts/data_smoke_test.py --config configs/cde_ff49.yaml
python scripts/data_smoke_test.py --config configs/cde_ff100.yaml

# Optional: one training batch's CDE forward/loss/backward, with no parameter update.
OMP_NUM_THREADS=1 python scripts/data_smoke_test.py --config configs/cde_ff49.yaml --model-smoke --smoke-batch-size 1 --device cpu
OMP_NUM_THREADS=1 python scripts/data_smoke_test.py --config configs/cde_ff100.yaml --model-smoke --smoke-batch-size 1 --device cpu
```

Add another source by returning a canonical panel from `data/sources/`, adding it
to the small `data/registry.py` dispatch, and selecting its dataset/options and RF
convention in config. Models, standardization, windows and diffusion run logic
do not need dataset-specific changes. Saved scalers still validate asset labels
and ordering; checkpoints and scalers from different asset panels are not interchangeable.

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

`scripts/diagnose_diffusion.py` checks timestep prediction error and stochastic
versus zero-posterior-noise reverse-chain scale on training examples only:

```bash
python scripts/diagnose_diffusion.py --config outputs/cde/config.yaml --checkpoint outputs/cde/best.pt --standardizer outputs/cde/standardizer.npz
```

It supports the existing Gaussian epsilon and v parameterizations and is a
sampler diagnostic, not a forecast benchmark. Its default noise seeds are
preserved from the original diagnostic. The historical router uses interleaved
validation noise, while diffusion training uses separate timestep/noise streams.
Routing recognizes both checkpoint conventions, keeps its original paired
comparison stream, and records a distinct saved checkpoint loss when needed.

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
