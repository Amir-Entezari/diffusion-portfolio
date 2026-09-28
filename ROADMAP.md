# Research Roadmap

## Current status

**Current stage: Stage 1 — Repository refactor**

The repository is being treated as an untrusted research scaffold. Passing unit tests do
not validate the scientific claims of the proposal.

## Stages

- [x] **0. Audit & reset** — map the generated code to the proposal; identify semantic failures.
- [ ] **1. Refactor skeleton** — package layout, config, scripts, semantic-test framework.
- [ ] **2. Data pipeline** — Kenneth French 12 Industry Portfolios, chronological splits, train-only normalization.
- [ ] **3. Evaluation first** — portfolio backtest + distribution metrics.
- [ ] **4. Classical baselines** — equal weight, rolling MVO, rolling mean-CVaR/min-var as appropriate.
- [ ] **5. Vanilla diffusion MVP** — conditional joint next-return scenarios.
- [ ] **6. Literature baseline** — reproduce/approximate a credible diffusion+portfolio baseline.
- [ ] **7. Proposal ablations I** — CDE, covariance geometry, TDA one at a time.
- [ ] **8. Fusion research** — concat -> attention -> mathematically valid SGW if justified.
- [ ] **9. Jump/tail dynamics** — controlled ablation only after vanilla diffusion works.
- [ ] **10. MSB decision** — resolve physical-vs-risk-neutral measure before implementation.
- [ ] **11. Dynamic controller** — only after multi-step scenarios are validated.
- [ ] **12. Deep BSDE / TTSA** — re-derive and implement only if justified.
- [ ] **13. Full experiments** — seeds, ablations, regimes, costs, robustness.
- [ ] **14. Proposal rewrite** — final research story follows evidence.

## MVP success criterion

A reproducible Kaggle experiment on Kenneth French 12 Industry Portfolios that compares
vanilla conditional diffusion against classical portfolio baselines using one shared evaluator.

## Explicitly inactive during MVP bring-up

The generated SGW, jump/MSB engine, Deep-BSDE controller, and TTSA code are **not** in
the active package. The original versions remain in git tag `matin-scaffold-v0`.
