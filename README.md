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
asset count from that panel. Controlled synthetic factor markets use the same
pipeline; the older experimental LOB/jump generator remains isolated and unused
by source dispatch. Baseline-specific objectives and adapters belong together
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
synthetic arithmetic-return source sets it to `false`; RF is then never loaded.
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

## Controlled synthetic markets

`synthetic_gaussian`, `synthetic_regime`, and `synthetic_regime_jump` load through
the same `load_return_table` / train-only normalization / causal window pipeline.
Matching `configs/<dataset>.yaml` files retain Vanilla diffusion settings and use
`subtract_risk_free: false`. No forecasting models are part of the simulator.
The [demonstration notebook](notebooks/synthetic/synthetic_market_validation.ipynb)
generates observations, saves resolved parameters and training-only diagnostics,
checks the shared pipeline, and provides separate CPU smoke / optional pilot cells.

The specified construction is
`x = mu[z] + sigma[z] * (B[z] @ f + D[z] @ epsilon) - J * a * b[z]`, with independent
standard Gaussian factor/idiosyncratic shocks. `x` is a **daily log return**;
`ReturnTable` receives **arithmetic decimal** `expm1(x)`. The no-jump covariance is
`sigma[z]**2 * (B[z] @ B[z].T + diag(d[z]**2))`, PSD by construction. Fixed jumps
add conditional mean `-p*a*b` and covariance `p*(1-p)*a**2*outer(b,b)`.
Prices are `initial_price * exp(cumsum(x))`; the first price is post-return,
with no artificial zero observation or price reset. Unrepresentable returns/prices
raise an error; they are never clipped. Prices can be omitted for long simulations.
"Gaussian" refers to log returns; the arithmetic returns have the transformed law.

Illustrative daily assumptions: normal drift 0.0002, first-factor loadings
0.004–0.008 (additional deterministic cosine exposures), idiosyncratic standard
deviation 0.01, multiplier 1. Stress uses drift -0.0004, doubled factor loadings,
idio 0.0075 and multiplier 1.5. Relative factor/idio exposure changes correlation;
scaling volatility alone does not. Markov rows `[[.99,.01],[.04,.96]]` imply
stationary occupancy 80%/20% and mean durations 100/25 weekdays. Initial state
defaults to that stationary distribution. Jump probabilities are **daily Bernoulli
probabilities**, 0.002/0.03, with fixed positive log magnitude 0.04 and nonnegative
asset exposures (default one); these are not Poisson intensities or fitted estimates.
A common negative jump term need not make every net observed return negative.

The 8,000 business weekdays span 1990-01-01–2020-08-28, with cutoffs derived at
observations 5,000 (2009-02-27) and 6,500 (2014-11-28). Usable windows are
**4,940 train / 1,500 validation / 1,500 held-out**. Only training dates enter
notebook diagnostics; held-out availability is counted, never used for selection.
Ground truth (regimes, jump events/magnitudes, volatility states) stays separate
from the ordinary source panel and model inputs. The source seed in `source_options`
is independent of the training seed. Four isolated PCG64 streams preserve prefixes
when extending sample length; save resolved parameters, simulator/NumPy versions
and code revision for exact reproduction. See NumPy's
[RNG version policy](https://numpy.org/doc/stable/reference/random/generator.html).

A stronger systemic-stress setting uses the **same** regime engine: for example
`n_factors: 1`, `factor_loadings: [[[0.01]], [[0.02]]]`,
`idiosyncratic_vol: [[0.006], [0.004]]`, `volatility_multipliers: [1.5, 2.5]` in
`source_options`. Scalars/asset vectors broadcast; per-regime drift/idio arrays
use `[regimes, assets]` (or `[regimes, 1]`). No separate stress model is introduced.

```bash
python scripts/data_smoke_test.py --config configs/synthetic_gaussian.yaml --model-smoke --device cpu
python scripts/data_smoke_test.py --config configs/synthetic_regime.yaml
python scripts/data_smoke_test.py --config configs/synthetic_regime_jump.yaml
```

Monte Carlo tests check analytical log moments, transition/duration uncertainty,
jump rates, return/price consistency and leakage. These checks validate a specified
simulator, **not empirical market realism**. Regime/jump mixtures are not
unconditionally Gaussian. The weekday calendar omits exchange holidays; there is
no stochastic-volatility recursion. Separate mathematical topology fixtures,
alpha-stable innovations, 4D irregular/intraday/event simulation and broader
downstream evaluation remain unimplemented and require separate scientific decisions.

### Extended synthetic stress scenarios (4C)

Three additional sources/configs use the **same** factor simulator, calendar,
partitions, return conversion and Vanilla training settings. Define the centered
Gaussian driver `v[t] = sigma[z[t]] * (B[z[t]] @ f[t] + D[z[t]] @ epsilon[t])`,
with covariance `Sigma[z]` as above. All coefficients are daily log-return units;
AR coefficients and Student-t scales are dimensionless. No new scenario has jumps.

- `synthetic_student_t`: `x[t] = mu + v[t] * sqrt((nu-2)/W[t])`, with **one**
  independent `W[t] ~ chi-square(nu)` shared by all factor/idiosyncratic shocks
  that day. The default is `nu=8`; `degrees_of_freedom` must be finite and `>4`.
  Log covariance stays `Sigma`, and nondegenerate marginal log excess kurtosis is `6/(nu-4)`.
  Variance exists for `nu>2`, fourth moments for `nu>4`; eighth moments need
  `nu>8`, so kurtosis can converge slowly at the default. A fifth isolated RNG
  child supplies scales; existing stream consumption is unchanged. This is the
  covariance-standardized elliptical [Student-t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html),
  not independent asset-wise rescaling. **Arithmetic returns `expm1(t)` have no
  finite positive population moments** because t has polynomial upper tails,
  despite finite observed samples; moment
  comparisons and tail thresholds explicitly use log returns.
- `synthetic_mean_reverting`: `y[0]=v[0]`, then
  `y[t]=phi*y[t-1]+sqrt(1-phi**2)*v[t]`, `x[t]=mu+y[t]`. Fixed Gaussian regime,
  default `phi=0.4`, strict `abs(phi)<1`. Stationary Gaussian initialization gives
  mean `mu`, covariance `Sigma`, and lag-k log covariance `phi**k * Sigma`.
  The actual AR shock covariance is `(1-phi**2)*Sigma`. This is return mean
  reversion, not an OU price-level model; prices continue compounding.
  Legacy `conditional_log_*` metadata conditions on regime only; it describes
  marginal moments, not the conditional forecast given observed history.
- `synthetic_whipsaw`: `y[t]=phi[z[t]]*y[t-1]+sqrt(1-phi[z[t]]**2)*v[t]`,
  `x[t]=mu[z[t]]+y[t]`. Default normal/whipsaw coefficients are `[0.1,-0.65]`,
  with the existing 4B Markov/driver settings. Both coefficients must have
  magnitude below one, and whipsaw must be negative and below normal. Initial
  deviation is Gaussian under the initial state, **not the full switching
  stationary law**. Driver and actual shock covariances are recorded separately;
  resulting regime-conditional return covariance is `None`, never silently
  equated to `Sigma[z]`. Empirical lag/reversal pairs stay within uninterrupted
  spells, including intermediate dates. Conditional AR slopes do not imply
  exact stationary correlations during switching. Raw sign reversals and
  reversals about the known log mean are reported separately.

`innovation_distribution`, `degrees_of_freedom` and `ar_coefficients` are explicit
source options. Only the combinations above are enabled; unsupported heavy-tail
AR/jump mixtures fail. Ground truth remains separate from model inputs. Finite
threshold co-exceedances are not asymptotic tail-dependence estimates. These
checks establish specified mechanisms, not empirical realism or forecasting gains.

The [4C Kaggle notebook](notebooks/synthetic/synthetic_market_stage4c_validation.ipynb)
uses `feature/synthetic-market`, training-only diagnostics, direct CPU smoke
commands and optional separate observation/ground-truth exports with source SHA.
The completed 4A/4B notebook is unchanged. Six raw-array hash fixtures captured
**before** the extension check the original scenarios at default/custom settings;
the fixture records its NumPy/Pandas environment for bitwise reproduction.
Log returns, dates, labels and ground-truth arrays always require exact hashes.
Returns and prices also require exact hashes under the reference NumPy 2.4.0;
other versions check their mathematical reconstruction to tight float64 tolerances.
Bitwise verification of these two derived arrays requires the reference NumPy version.

```python
from diffusion_portfolio.data.registry import load_return_table
table = load_return_table("synthetic_student_t", degrees_of_freedom=8)
table = load_return_table("synthetic_mean_reverting", ar_coefficients=0.4)
table = load_return_table("synthetic_whipsaw", ar_coefficients=[0.1, -0.65])
```

```bash
python scripts/data_smoke_test.py --config configs/synthetic_student_t.yaml --model-smoke --device cpu
python scripts/data_smoke_test.py --config configs/synthetic_mean_reverting.yaml --model-smoke --device cpu
python scripts/data_smoke_test.py --config configs/synthetic_whipsaw.yaml --model-smoke --device cpu
pytest -q tests/unit/test_synthetic_market_stage4c.py tests/unit/test_synthetic_market.py tests/unit/test_synthetic_market_pipeline.py
python -m compileall src scripts
pytest -q
```

## Controlled synthetic forecast baselines

`scripts/benchmark_synthetic_forecasts.py` compares three joint Gaussian forecasts
on identical validation targets. It fits unique chronological training observations
in observable log space, `x=log1p(r)`, samples joint log returns, then scores
`expm1(x)` arithmetic decimal scenarios with the existing CRPS, Energy Score and
interval calibration implementation. Hidden simulator state is never an input.

- **Gaussian:** training MLE mean and covariance, with divisor T; parameters freeze
  at `train_end`.
- **AR(1):** `x_next=c+phi*x+epsilon`, one pooled scalar phi and asset-specific
  intercepts. OLS demeans training predictors/responses separately; stationary
  mean is `mu=c/(1-phi)`. Nonstationary fits (`abs(phi)>=1`) fail rather than being
  clipped. Covariance is the MLE of fitted one-step residuals, not marginal return
  covariance. Training pairs spanning long calendar gaps are excluded.
- **EWMA:** initialize from the training MLE at `train_end`, then update each newly
  observed log return once, with fixed default decay lambda=0.94:
  `d=x-m_old`, `m_new=m_old+(1-lambda)*d`,
  `C_new=lambda*C_old+lambda*(1-lambda)*d*d.T`. This is the centered covariance
  of an exponential mixture and preserves PSD. Earlier validation observations
  may update the state; the current target never enters its own forecast.

Gaussian/AR fits use all original float64 training rows, including the initial
lookback, without duplicating overlapping windows. AR prediction uses the last
observable return in the shared raw history window. EWMA consumes all available
post-training source observations strictly before each target, including dates
omitted by window segmentation. Scored targets retain the shared pipeline's
float32 representation; scenarios are float64 arithmetic returns.

Sampling uses a PSD eigensquare root without a ridge. Only negative eigenvalues
within `100*float64_eps*spectral_scale` of zero are corrected; materially indefinite
covariances fail. Invalid returns (`r<=-1`), nonfinite values and unrepresentable
samples fail; return tails are never clipped. Separate PCG64 streams keyed by
seed, fixed model ID and target-date ordinal make model order and validation-limit
changes preserve forecasts for common dates in the same numerical environment.

```bash
python scripts/benchmark_synthetic_forecasts.py \
  --config configs/synthetic_gaussian.yaml \
  --models gaussian,ar1,ewma --split validation \
  --n-scenarios 128 --limit 64 --seed 42 \
  --output-dir outputs/synthetic_benchmarks/gaussian_pilot

python scripts/benchmark_synthetic_forecasts.py \
  --config configs/synthetic_mean_reverting.yaml \
  --models gaussian,ar1,ewma --split validation \
  --n-scenarios 128 --limit 64 --seed 42 \
  --output-dir outputs/synthetic_benchmarks/ar1_pilot

pytest -q tests/unit/test_statistical_baselines.py tests/unit/test_synthetic_forecast_benchmarks.py
```

`--limit` selects the first N available validation targets; omit it for all
validation dates. Output directories must be empty or absent. Each run writes
resolved `config.yaml`, train-fitted `standardizer.npz`, `report.json`,
`comparison.csv`, and per-model `validation_scenarios.npz` (existing keys:
`scenarios`, `observed`, `dates`, `columns`) and probabilistic metric JSON.
Reports include source/config fingerprints, fitted parameters, seeds, dates,
units, runtime, per-asset CRPS and absolute calibration error. The scaler is
saved for pipeline compatibility; statistical fits use unstandardized logs.

Supported sources are Gaussian, mean-reverting, regime, regime-jump and whipsaw.
**Student-t is rejected** because exponentiation gives infinite positive
population moments; its log-space diagnostics remain available. Config-aware
diffusion/portfolio scoring also rejects this source. Array-only metric functions
cannot infer source identity. No held-out test evaluation or diffusion training
is included in the statistical runner. The paired Vanilla workflow below verifies
source/configuration, target-date, asset-order and observation matches.

The [CPU Kaggle demo](notebooks/synthetic/synthetic_forecast_benchmarks.ipynb)
runs small Gaussian and AR pilots, displays saved metrics and a preselected example
distribution. Finite-ensemble scores have Monte Carlo error; small pilots do not
establish significance. Multi-seed uncertainty analysis is deferred. These
comparisons may favor correctly specified statistical models; no diffusion
advantage is presumed.

## Paired synthetic Vanilla comparisons

`evaluate_diffusion.py --limit N --output-dir <new-directory>` samples only the
first N validation windows, slicing before the DataLoader. Limited inference
requires validation and a separate empty/absent destination. Omitting the new
options preserves existing full evaluation, RNG conventions and output filenames.
Limits larger than the available validation count are rejected, not silently capped.
The configured scenario count (256 for the synthetic configs) is retained.

New Gaussian/mean-reverting Vanilla training records observation-based
`data_identity.json` beside existing checkpoints. It hashes the train/validation
panel and simulator source and records data settings, dates and assets. Before
training starts, the same sidecar freezes the training contract: seed,
model, AdamW identity, all training settings (including the original epoch budget)
and evaluation settings. Resume, evaluation and comparison reject changes to this
contract. Runtime device/resume/stopping options are excluded, so stopping after
two epochs can resume to the unchanged configured 50-epoch budget. Existing
checkpoints lacking this contract are not automatically upgraded or certified;
their historical resume and default evaluation remain available, while strict
comparison requires the original recorded contract.
These additions are metadata only: checkpoint schema, scaler, model, optimizer and training RNG are
unchanged. After a completed training chunk, the sidecar also binds source
identity to the exact best/last checkpoint hashes. A chunk interrupted before
this record is written fails strict verification until safely resumed; the
tool never certifies mismatched checkpoint bytes from filenames or losses alone.
Legacy runs remain usable by the original evaluator; missing source
evidence is never retrospectively manufactured. Strict comparisons reject such
runs rather than certify an unverifiable training source.

Execution pilot, using the original 50-epoch configuration with an absolute
two-epoch stopping boundary:

```bash
python -u scripts/train_diffusion.py --config configs/synthetic_gaussian.yaml \
  --output-dir outputs/synthetic_diffusion/gaussian_execution_pilot \
  --device cuda --resume --stop-after-epoch 2

python -u scripts/evaluate_diffusion.py \
  --run-dir outputs/synthetic_diffusion/gaussian_execution_pilot \
  --split validation --limit 64 --batch-size 16 --device cuda \
  --output-dir outputs/synthetic_comparisons/gaussian_pilot_inference

python scripts/compare_synthetic_forecasts.py \
  --baseline-dir outputs/synthetic_benchmarks/gaussian_full \
  --diffusion-dir outputs/synthetic_comparisons/gaussian_pilot_inference \
  --limit 64 --block-length 20 --n-bootstrap 1000 --bootstrap-seed 42 \
  --output-dir outputs/synthetic_comparisons/gaussian_pilot_comparison
```

An existing full 1,500-date Stage 4E-1 baseline directory is reusable without
rerunning it. If unavailable, generate a matched small baseline with
`benchmark_synthetic_forecasts.py --config <pilot-run>/config.yaml --n-scenarios 256
--limit 64 --output-dir <new-directory>`. A 128-scenario pilot is not interchangeable
with a 256-scenario diffusion forecast. Model-specific sampling seeds may differ.

Inference copies source/config/scaler evidence and writes `forecast_metadata.json`
with artifact hashes, checkpoint epoch/loss, declared/completed epoch counts,
preliminary/completed-budget status, dates, units, seed, batch size, device and
library versions. It verifies the saved scaler against reconstructed train-only
moments without refitting. `last.pt` history takes precedence over possibly stale
CSV history. A partial run is preliminary; full-budget status also requires a
consistent summary. Missing or inconsistent metadata fails clearly.

Comparison reconstructs canonical validation targets, verifies data settings and
source fingerprints, checks exact observations/assets/dates and equal ensemble
size, then uses the matching baseline prefix. Nothing is reordered, imputed,
clipped or refitted. Numerical-library changes that alter source fingerprints
must be investigated; source mismatches are not silently tolerated. New sampling
is reproducible with identical settings in one environment, but batch size,
last-batch size and CPU/GPU differences may change individual diffusion draws.

Outputs are `comparison_report.json`, `per_date_scores.csv`,
`paired_comparison.csv` and separate `calibration.csv` (50/80/90/95/99% coverage
and absolute errors). Per-date CRPS and Energy Score use exactly the historical
ensemble estimators. Differences are **diffusion minus baseline** (negative favors
diffusion); reports retain both score means, absolute differences, relative
improvement fractions and sample standard deviations of paired differences.

The percentile 95% moving-block bootstrap samples overlapping non-circular
contiguous blocks with replacement, truncates to the original series length and
uses the same resampled dates for every paired series. Defaults are block length
20 forecast observations, 1,000 replications and seed 42. Blocks longer than the
series are rejected. It assumes approximately stationary differences and adequate
block length; short pilots are exploratory. It conditions on saved ensembles and
does not capture repeated Monte Carlo, training or checkpoint-selection uncertainty.

**Validation selected the checkpoint. These comparisons are development evidence,
not independent final generalization estimates or selection-adjusted significance.**
The [Kaggle pilot notebook](notebooks/synthetic/synthetic_diffusion_comparison.ipynb)
runs the two-epoch Gaussian workflow, accepts user-supplied baseline directories,
and provides full training commands as unexecuted Markdown. To start meaningful
training manually, use the original Gaussian/mean-reverting configs with separate
`gaussian_full` / `mean_reverting_full` run directories and `--resume`, without the
stopping flag. Do not infer forecast quality merely from completing 50 epochs.
The held-out test partition remains untouched.

## Controlled historical-context ablation

The [context notebook](notebooks/synthetic/synthetic_history_ablation.ipynb)
studies `synthetic_mean_reverting` with K0/K1/K5/K60 available observations.
`configs/ablations/synthetic_ar1_context.yaml` changes **only** the original
mean-reverting configuration's epoch budget from 50 to a predeclared 15.
Every arm retains the 60-day window, 12 assets, source seed, training seed,
targets, splits, scaler, MLP shape/parameter count, optimizer and v-prediction
objective. Fifteen epochs is a diagnostic budget, not a tuned optimum.

`--history-keep-last K` attaches a parameter-free encoder pre-hook. It replaces
earlier **standardized** history with zeros (the fitted training mean), without
mutating inputs or masking targets. The same hook runs during training,
validation loss and sampling. K60 is exactly the unmasked numerical path.
The optional mask is recorded in the immutable training contract, with original
lookback, effective history and convention; historical contracts gain no fields.
A small ablation-only `experiment.json` marker also rejects missing policy
evidence. Resume must specify the original K; evaluation restores it automatically.
No checkpoint tensor keys change, and no legacy checkpoint is certified or migrated.
Keep both provenance files with the checkpoints and inference outputs.

Two-epoch execution pilot (repeat the commands with K60 and its separate directory):

```bash
python -u scripts/train_diffusion.py \
  --config configs/ablations/synthetic_ar1_context.yaml --history-keep-last 1 \
  --output-dir outputs/synthetic_context/k1 --device cuda --resume --stop-after-epoch 2
python -u scripts/evaluate_diffusion.py --run-dir outputs/synthetic_context/k1 \
  --split validation --limit 64 --batch-size 16 --device cuda \
  --output-dir outputs/synthetic_context/k1_pilot_inference
python scripts/compare_synthetic_forecasts.py \
  --baseline-dir <existing-compatible-mean-reverting-baselines> \
  --diffusion-dir outputs/synthetic_context/k1_pilot_inference \
  --block-length 20 --n-bootstrap 1000 --bootstrap-seed 42 \
  --output-dir outputs/synthetic_context/k1_pilot_comparison
```

Existing full statistical archives remain reusable with unchanged source, dates,
observations, asset order and 256 scenarios. Otherwise generate a small matched
baseline with `benchmark_synthetic_forecasts.py --config
configs/ablations/synthetic_ar1_context.yaml --n-scenarios 256 --limit 64
--output-dir <new-baseline-directory>`. Statistical models receive actual
unmasked observable history. The source/alignment checks are unchanged.

For the matched 15-epoch study, use the same training command **without**
`--stop-after-epoch`, independently for `--history-keep-last 0`, `1`, `5` and `60`
with `k0`, `k1`, `k5`, `k60` directories. K1/K60 resume their pilots to the
original 15-epoch budget. Evaluate each best checkpoint into a new directory
(omit `--limit` for all validation dates), and run the paired baseline tool per arm.
Do not mix the historical 50-epoch K60 run into this primary comparison.

```bash
python scripts/diagnose_history_ablation.py --baseline-dir <baseline-directory> \
  --diffusion-dirs K1=outputs/synthetic_context/k1_pilot_inference K60=outputs/synthetic_context/k60_pilot_inference \
  --block-length 20 --n-bootstrap 1000 --bootstrap-seed 42 \
  --output-dir outputs/synthetic_context/pilot_diagnostics
pytest -q tests/unit/test_history_ablation.py tests/unit/test_synthetic_diffusion_comparison.py
```

The archive-only diagnostic verifies original contracts and exact target alignment;
add K0/K5 archives for all six paired arm comparisons. It averages `log1p` scenarios
to estimate predictive log means, demeans history and predictive means **per asset**
across dates before pooling the lag slope, and reports predictive log-mean MSE.
CRPS/Energy Score/calibration retain their arithmetic-return definitions. It saves
`summary.csv`, `per_date_scores.csv`, `paired_context.csv`, `calibration.csv` and
`diagnostics.json`, including best epoch/loss, final losses, completed budget and
parameter count. Arm contracts, dates, scenario counts and completed budgets must
match exactly. No automatic intersection or historical-budget substitution occurs.

K0 showing weak direct lag dependence, K1 helping a one-lag process, and longer
contexts complicating learning are **hypotheses**. Primary outcomes are CRPS,
Energy Score and predictive-mean MSE; lag slope and train/validation divergence
are explanatory diagnostics. Means are finite-ensemble estimates. Paired block
intervals condition on those ensembles and do not correct validation checkpoint
selection. No held-out test evaluation is included. The notebook executes only
K1/K60 two-epoch pilots by default; full-budget commands remain manual Markdown.

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
