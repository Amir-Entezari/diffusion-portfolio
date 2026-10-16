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

Four self-contained Kaggle notebooks under `notebooks/high_dimensional/` cover
FF49/FF100 with Vanilla MLP and Neural CDE conditioning. The Vanilla configs are
`configs/vanilla_ff49.yaml` and `configs/vanilla_ff100.yaml`: their model, optimizer
and evaluation settings match `mvp.yaml`, and their data settings match the paired
CDE configs. Existing KF12 notebooks and configs are unchanged.

Enable Kaggle Internet and a GPU. The notebooks use ordinary interactive cells:
run setup/tests/restore, inspect a two-epoch pilot, then manually resume to epoch
50 in the **same run directory**. `BRANCH="feature/high-dimensional-kaggle"` is
used until merging; change that one variable to `main` afterwards. Seed-42 runs
remain at `/kaggle/working/ff49_vanilla`, `ff49_cde_clean`, `ff100_vanilla`, and
`ff100_cde`. The old damaged `ff49_cde` artifact is never used or repaired.
Training prints one flushed loss/timing line per completed epoch. Plots prefer
`last.pt` history over a potentially stale CSV. Commands check their exit status.

The following matrix maps the retained KF12 methods to high-dimensional runs.
**Ready** means implementation/CLI/fixture checks passed, not that an experiment
has run or produced a positive finding. New downstream notebooks fail early
until their prerequisite seed-42 models finish. All comparisons are validation
only; FF100 remains exploratory because of its 68 validation windows.

| Dataset | Experiment / notebook | Prerequisites | Expected artifacts | Validation status | Implementation |
|---|---|---|---|---|---|
| FF49 | [Vanilla](notebooks/high_dimensional/ff49_vanilla.ipynb), [CDE](notebooks/high_dimensional/ff49_neural_cde.ipynb) | None | config/scaler, best/last, history, summary, validation metrics/scenarios | Data/model smoke and notebook tests | Existing; smoke now, training manual |
| FF100 | [Vanilla](notebooks/high_dimensional/ff100_vanilla.ipynb), [CDE](notebooks/high_dimensional/ff100_neural_cde.ipynb) | None | Same, isolated FF100 runs | Data/model smoke and notebook tests | Existing; exploratory |
| FF49 | [Multi-seed](notebooks/high_dimensional/ff49_multiseed.ipynb) | Completed Vanilla + CDE seed 42 **and validation metrics** | Resumable seeds 43/44; per-seed and paired mean/std tables | Offline CLI, source/isolation and aggregation tests | Ready after prerequisites |
| FF100 | [Multi-seed](notebooks/high_dimensional/ff100_multiseed.ipynb) | Same, FF100 references | Same under FF100 namespace | Same; no forecasting results asserted | Ready; exploratory |
| FF49 | [Evidential/routing](notebooks/high_dimensional/ff49_evidential_router.ipynb) | Completed CDE + validation metrics; router needs current-regime probe | Next-day/current heads, thresholds, predictions; softmax/evidential router comparisons | Dimension, train-threshold, source and CLI checks | Ready after prerequisites; single-shot steps |
| FF100 | [Evidential/routing](notebooks/high_dimensional/ff100_evidential_router.ipynb) | Same, FF100 references | Same under FF100 namespace | Same; small validation sample caveat | Ready; exploratory |
| FF49 | [TDA](notebooks/high_dimensional/ff49_tda.ipynb) | Completed frozen CDE | Validated feature cache; CDE/geometry/TDA best checkpoints and metrics | Tiny feature/normalization/date/gradient and notebook checks | Ready after prerequisite + feasibility gate |
| FF100 | [TDA](notebooks/high_dimensional/ff100_tda.ipynb) | Completed frozen CDE | Same under FF100 namespace | Same; no full extraction executed | Ready; exploratory |
| FF49 | [SPD](notebooks/high_dimensional/ff49_spd.ipynb) | Completed frozen CDE | Covariance/Log-SPD cache, CDE/covariance/Log-SPD checkpoints and metrics | 1,225-dimensional fixture checks + feasibility gate | Ready after prerequisite + feasibility gate |
| FF100 | [SPD](notebooks/high_dimensional/ff100_spd.ipynb) | Completed frozen CDE | Same; 5,050-dimensional covariance features | Rank-deficient covariance fixture checks; real GPU feasibility pending | Ready; exploratory, feasibility required |
| FF49 | Diffolio (no runnable notebook claimed) | Approved source/warmup/continuity protocol; Goyal workbook | None created | Existing KF12 implementation inspected | **Blocked** |
| FF100 | Diffolio (no runnable notebook claimed) | Same, plus policy for rolling characteristics across long gaps | None created | Existing KF12 implementation inspected | **Blocked** |

New notebooks use `/kaggle/working/ff{49,100}_{multiseed,evidential_router,tda,spd}`
and matching `high_dimensional/<directory-name>` namespaces. They restore only
the needed references and their own outputs. Manual backup first downloads the
latest complete central dataset, preserves unrelated experiments, and refuses
failed/empty downloads. **Serialize uploads across sessions**: there is no
cross-session lock, so simultaneous manual uploads can replace each other's
new versions. Nothing uploads automatically during setup or smoke checks.
Keep the cached official source ZIPs alongside saved artifacts; French can
revise historical observations. Source fingerprints bind caches and new probes
and routers to the exact saved CDE config, scaler and best checkpoint.

Multi-seed adds `--vanilla-config` and `--cde-config` to `run_multiseed.py`;
omitting them keeps the KF12 defaults. Seed 42 is never retrained. Additional
runs preserve their saved state and skip completed validation. Feature
extraction adds `--smoke` (two training histories, finite/dimension/spectral and
cache-memory report) and `--reuse` (verify source identity, exact prepared returns/dates and train-only
normalization before reusing caches). Neither flag changes default KF12 feature
mathematics. TDA/SPD ablations use the existing resumable shared trainer and
retain the original full feature dimensions and relative eigenvalue floor.
The covariance rank bound is 59 at lookback 60: the FF100 feasibility check must
pass without silently changing dtype, regularization or dimensionality. Cache
estimates exclude substantial additional return-window, concatenation and GPU
memory. No full extraction or GPU feasibility result is claimed here.

Probe and router CLIs remain **single-shot**; their notebook cells refuse to
overwrite existing directories and do not invent resume/pilot arguments.
Completed substeps can be restored and inspected without rerunning training.
The probes preserve the historical train-only scaler refit; the notebook checks
it against the saved CDE scaler first. Routing uses only the current-regime
heads, train-median vacuity and the retained interleaved validation corruption
stream. That control loss is reported separately from a CDE checkpoint trained
with separate validation timestep/noise streams. ECE/NLL/Brier classification
diagnostics are distinct from forecast CRPS/Energy Score/ACE.

**Diffolio blockers:** the model/ten characteristic channels/eight systematic
covariates are largely parameterized, but `prepare_diffolio_data` does not pass
French missing-row source options and its 63-observation windows do not use the
shared continuity guard. Characteristics require 756 observations before the
first requested date; FF49 starts too late for the unchanged 1958 request.
The rolling momentum/beta/idiosyncratic-volatility calculations also currently
count retained rows across FF100's multi-year gaps. A segment-aware 756-observation
warmup after the July 2004 restart would leave no validation window before
2004-12-31 under the current boundaries. A segment/warmup policy and
its effect on sample availability must be explicitly selected and validated;
changing dates or filling returns here would silently change the reproduction.
The original batch-1024/100k-step architecture's memory requirements at 100
assets are also unverified. No Diffolio config or misleading runnable notebook
is fabricated, and the KF12 reproduction is unchanged.

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
