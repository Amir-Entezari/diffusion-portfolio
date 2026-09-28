"""
Master Configuration for the Quantum-Topological Generative Diffusion Framework.

Every hyperparameter referenced in the dissertation proposal is encoded here
with its exact default value and a docstring mapping it to the relevant
section and equation in the proposal.

All configs are pure Python ``@dataclass`` objects so they can be:
    1. Instantiated programmatically in code.
    2. Serialised / deserialised to/from YAML via ``dataclasses.asdict``.
    3. Composed into the ``MasterConfig`` which is the single source of truth
       passed to every module in the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
#  Phase 0: Dynamic Evidential Router (§5.2)
# ---------------------------------------------------------------------------
@dataclass
class Phase0Config:
    """Configuration for the Neural CDE + Dirichlet Evidential Router.

    Reference: Proposal §5.2 — Dynamic Evidential Router based on Neural CDE
    and Dirichlet Distribution.

    The Neural CDE evolves the hidden state via:
        dH(t) = f_θ(H(t))dt + g_θ(H(t))dX(t)         (Eq. §5.2.1)
    where f_θ and g_θ are parameterised MLPs.

    The Dirichlet head maps H(t) → α(t) → u(t):
        e(t) = Softplus(W_ev · H(t) + b_ev)            (Eq. §5.2.2)
        α_k(t) = e_k(t) + 1
        u(t)   = K / Σ α_k(t)

    The phase-transition gate:
        G_k(t) = [1 − σ(γ(u−τ))]·p_k + σ(γ(u−τ))·1/K  (Eq. §5.2.3)
    """

    # --- Neural CDE core ---
    hidden_dim: int = 256
    """h: Dimension of the continuous hidden state H(t) ∈ ℝ^h."""

    drift_n_layers: int = 3
    """Number of hidden layers in the autonomous drift network f_θ."""

    drift_hidden_dim: int = 512
    """Hidden dimension of each layer in f_θ."""

    drift_activation: str = "tanh"
    """Activation for f_θ.  Tanh chosen for bounded outputs (Lipschitz)."""

    sensitivity_n_layers: int = 2
    """Number of hidden layers in the stochastic sensitivity network g_θ."""

    sensitivity_hidden_dim: int = 256
    """Hidden dimension of each layer in g_θ."""

    # --- ODE Solver ---
    solver: str = "dopri5"
    """Adaptive ODE solver for torchdiffeq (Dormand-Prince 5(4))."""

    rtol: float = 1e-4
    """Relative tolerance for adaptive stepping."""

    atol: float = 1e-5
    """Absolute tolerance for adaptive stepping."""

    use_adjoint: bool = True
    """If True, use odeint_adjoint for O(1) memory (§8.2.1).
    The adjoint sensitivity method replaces BPTT through the solver:
        da(t)/dt = −a(t)^T · ∂f_θ/∂H(t)
    collapsing memory from O(L·N) to O(1)."""

    # --- Dirichlet Evidential Head ---
    n_regimes: int = 4
    """K: Number of distinct market regimes
    (e.g., bull, bear, stochastic vol, flash crash)."""

    # --- Phase-Transition Gate ---
    uncertainty_threshold: float = 0.4
    """τ ∈ (0,1): Critical uncertainty threshold.
    When u(t) > τ, the gate collapses to uniform consensus 1/K."""

    gate_temperature: float = 10.0
    """γ ∈ ℝ⁺: Logistic temperature controlling sigmoid steepness.
    High γ → hard switch;  Low γ → smooth transition."""


# ---------------------------------------------------------------------------
#  Phase 1: Temporal Topology Extraction via TDA (§5.3)
# ---------------------------------------------------------------------------
@dataclass
class Phase1Config:
    """Configuration for Topological Data Analysis pipeline.

    Reference: Proposal §5.3 — Phase 1: Temporal-Topology Extraction via TDA.

    Pipeline:
        1. Takens delay embedding:
           Y(t) = [H(t)^T, H(t−τ)^T, ..., H(t−(m−1)τ)^T]^T  (Eq. §5.3.1)

        2. Vietoris-Rips filtration:
           VR(ε) = {σ ⊆ Y(t) | d(u,v) ≤ ε ∀ u,v ∈ σ}       (Eq. §5.3.2)

        3. Persistent homology → Betti numbers:
           β_k = rank(H_k) = dim(ker ∂_k) − dim(im ∂_{k+1})  (Eq. §5.3.3)

        4. Persistence landscape vectorisation:
           λ_k(t,ε) = k-max_i max(0, min(ε−b_i, d_i−ε))      (Eq. §5.3.4)
    """

    # --- Takens Embedding ---
    embedding_dim: int = 10
    """m: Embedding dimension.  Must satisfy m > 2d where d is the
    box-counting fractal dimension of the underlying market attractor
    (Whitney / Takens theorem)."""

    time_delay: Optional[float] = None
    """τ: Time-delay parameter.  If None, auto-selected via the first
    local minimum of the mutual information function of H(t)."""

    time_delay_max_lag: int = 50
    """Maximum lag to search when auto-selecting τ."""

    # --- Vietoris-Rips Filtration ---
    max_homology_dim: int = 2
    """Maximum homological dimension to compute.
    β_0 = connected components, β_1 = loops, β_2 = voids."""

    max_edge_length: float = 2.0
    """ε_max: Maximum filtration radius."""

    n_filtration_steps: int = 100
    """Number of discrete ε values in [0, ε_max]."""

    # --- Persistence Landscape ---
    n_landscape_layers: int = 5
    """Number of landscape layers (k-th largest envelopes)."""

    n_landscape_points: int = 100
    """Number of discretisation points for each landscape function."""

    # --- TDA Backend ---
    tda_backend: str = "gudhi"
    """Backend library: 'gudhi' or 'giotto-tda'."""

    use_gradient_checkpointing: bool = True
    """Apply gradient checkpointing to TDA extraction layers (§8.2.2).
    Trades ~20% extra compute for ~70% VRAM reduction."""


# ---------------------------------------------------------------------------
#  Phase 2: Quantum Information Geometry (§5.4)
# ---------------------------------------------------------------------------
@dataclass
class Phase2Config:
    """Configuration for Quantum Spatial Geometry extraction.

    Reference: Proposal §5.4 — Phase 2: Spatial-Relational Extraction
    using Quantum Information Geometry and the Bures Metric.

    Pipeline:
        1. Covariance → Density Matrix:
           ρ_t = Σ_t / Tr(Σ_t)                                (Eq. §5.4.1)

        2. Bures metric distance:
           D_B(ρ₁,ρ₂)² = 2(1 − Tr(√(√ρ₁ · ρ₂ · √ρ₁)))     (Eq. §5.4.2)

        3. Logarithmic map to tangent space:
           v_t = ρ_ref^{1/2} log(ρ_ref^{-1/2} ρ_t ρ_ref^{-1/2}) ρ_ref^{1/2}
                                                                (Eq. §5.4.3)
    """

    # --- Density Matrix ---
    cov_window: int = 60
    """Rolling window size (in ticks / time-steps) for computing
    the empirical covariance matrix Σ_t."""

    regularization_eps: float = 1e-6
    """ε: Tikhonov regularisation added to Σ_t before normalisation.
    Σ_t ← Σ_t + ε·I  ensures strict positive-definiteness."""

    # --- Bures Metric ---
    eigenvalue_clamp_min: float = 1e-8
    """Minimum eigenvalue clamp for numerical stability in matrix sqrt.
    Prevents NaN from near-zero eigenvalues during Black Swan convergence."""

    # --- Logarithmic Map ---
    reference_point: str = "identity"
    """Base point ρ_ref for the tangent space projection.
    Options: 'identity' (scaled identity), 'ema' (exponential moving average
    of market states), 'frechet_mean' (Fréchet mean on SPD manifold)."""

    ema_decay: float = 0.99
    """Decay factor for EMA reference point (only used if reference_point='ema')."""

    # --- Output ---
    flatten_tangent: bool = True
    """If True, flatten the symmetric tangent matrix v_t ∈ ℝ^{N×N}
    to its upper-triangular vector ∈ ℝ^{N(N+1)/2} for neural processing."""


# ---------------------------------------------------------------------------
#  SGW Interface: Sliced Gromov-Wasserstein (§5.5)
# ---------------------------------------------------------------------------
@dataclass
class SGWConfig:
    """Configuration for the SGW Manifold Alignment interface.

    Reference: Proposal §5.5 — Mathematical Interface: Manifold Alignment
    via Sliced Gromov-Wasserstein (SGW).

    The SGW distance:
        SGW(T_t, v_t) = ∫_{S^{d−1}} GW_1D(⟨T_t,θ⟩, ⟨v_t,θ⟩) dθ
                       ≈ (1/L) Σ_l GW_1D(...)                 (Eq. §5.5.2)

    SGW Cross-Attention:
        c_unified = Σ_k (Softmax(Q·K^T/√d_k) ⊙ π*) V        (Eq. §5.5.3)
    """

    # --- Sliced Projections ---
    n_projections: int = 50
    """L: Number of random projection vectors θ ∈ S^{d−1}.
    The integral over the hypersphere is approximated by averaging
    over L Monte-Carlo sampled directions."""

    projection_dim: int = 64
    """Shared embedding dimension for projecting both topological
    and quantum features before slicing."""

    # --- Cross-Attention ---
    n_heads: int = 8
    """Number of attention heads in SGW cross-attention."""

    attention_dim: int = 64
    """d_k: Per-head dimension for Q, K, V projections."""

    unified_dim: int = 256
    """D: Dimension of the output unified conditioning tensor c_unified."""

    dropout: float = 0.1
    """Dropout rate in the cross-attention module."""

    # --- Computational ---
    use_gradient_checkpointing: bool = True
    """Checkpoint the projection loop to reduce VRAM."""

    use_entropic_regularization: bool = False
    """If True, use entropic-regularised GW (Sinkhorn) instead of exact 1D sort.
    Generally not needed since 1D GW is already O(N log N)."""

    entropic_reg: float = 0.01
    """ε for Sinkhorn regularisation (only if use_entropic_regularization=True)."""


# ---------------------------------------------------------------------------
#  Phase 3: Jump-Diffusion MSB Engine (§5.6)
# ---------------------------------------------------------------------------
@dataclass
class Phase3Config:
    """Configuration for the Generative Jump-Diffusion Schrödinger Bridge engine.

    Reference: Proposal §5.6 — Phase 3: The Generative Jump-Diffusion Engine
    Constrained by Martingale Schrödinger Bridges (MSB).

    Forward SDE (§5.6.1):
        dY_t = f(Y_t,t)dt + g(t)dW_t + J(Y_{t⁻})dN_t

    Reverse SDE (§5.6.2):
        dY_t = [f − g²∇ log p_t − ∫ z λ_t(z) p(dz|Y_t)]dt + g d̄W_t + d̄J_t

    MSB Loss (§5.6.3):
        L_MSB = E[‖s_θ − ∇ log p_{0t}‖²] + η E[‖E_θ[P_{t−Δt}|P_t] − P_t‖₁]
    """

    # --- Diffusion Time ---
    n_diffusion_steps: int = 1000
    """T: Total number of diffusion steps (forward destruction)."""

    # --- Noise Schedule ---
    beta_start: float = 1e-4
    """β(0): Starting value of the variance schedule."""

    beta_end: float = 0.02
    """β(T): Terminal value of the variance schedule."""

    noise_schedule_type: str = "cosine"
    """Type of β(t) schedule: 'linear', 'cosine', or 'learned'."""

    # --- Jump Process ---
    jump_intensity: float = 0.05
    """λ: Base Poisson jump intensity (probability of jump per step).
    Represents frequency of Black Swan / flash crash events."""

    jump_distribution: str = "normal_inverse_gaussian"
    """Distribution family for jump magnitudes J(Y_{t⁻}).
    Options: 'normal_inverse_gaussian', 'variance_gamma', 'gaussian'."""

    jump_scale: float = 0.1
    """Scale parameter for the jump magnitude distribution."""

    # --- Score Network (UNet / Transformer) ---
    score_network_type: str = "unet_1d"
    """Architecture type: 'unet_1d' or 'transformer'."""

    unet_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    """Channel dimensions for each UNet resolution level."""

    unet_attention_resolutions: List[int] = field(default_factory=lambda: [2, 3])
    """Resolution levels (0-indexed) at which to apply self-attention."""

    time_embed_dim: int = 128
    """Dimension of the sinusoidal time embedding."""

    condition_embed_dim: int = 256
    """Dimension for c_unified condition injection (must match SGW unified_dim)."""

    n_residual_blocks: int = 2
    """Number of residual blocks per UNet resolution level."""

    use_adaptive_group_norm: bool = True
    """If True, inject c_unified via Adaptive Group Normalisation (AdaGN)."""

    # --- Martingale Schrödinger Bridge ---
    msb_penalty_weight: float = 10.0
    """η: Lagrange multiplier for the Martingale violation penalty.
    Higher η → stronger no-arbitrage enforcement at the cost of
    slower score-matching convergence."""

    risk_free_rate: float = 0.02
    """r_f: Continuous annual risk-free rate for discounting."""

    # --- Sampling ---
    n_sampling_steps: int = 250
    """Number of reverse SDE steps for generation (can be < n_diffusion_steps
    via accelerated sampling schemes)."""

    use_predictor_corrector: bool = True
    """If True, apply Langevin MCMC correction steps during sampling."""

    n_corrector_steps: int = 1
    """Number of Langevin correction steps per predictor step."""

    corrector_snr: float = 0.16
    """Signal-to-noise ratio for Langevin dynamics."""


# ---------------------------------------------------------------------------
#  Phase 4: Deep BSDE Controller (§5.7)
# ---------------------------------------------------------------------------
@dataclass
class Phase4Config:
    """Configuration for the Deep BSDE Portfolio Controller.

    Reference: Proposal §5.7 — Phase 4: Strategic Asset Allocation
    Controller (Deep BSDE) with Huber Loss Approach.

    Wealth process (§5.7.1):
        dX_t/X_{t⁻} = (1 − 1^T W_t)r_f dt + W_t^T dY_t − Γ(W_t,W_{t⁻})dt

    Huber friction (§5.7.2):
        Γ = c · Σ_i H_δ(W_{t,i} − W_{t⁻,i})

    Policy (§5.7.3):
        W_t = φ_ω(t, X_{t⁻}, Y_{t⁻}, c_unified)

    Outer loss (§6.3.2):
        L_outer = −E[ U(X_T) − c ∫ Σ H_δ(dφ/dt) dt ]
    """

    # --- Policy Network ---
    policy_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    """Hidden layer dimensions for the Deep BSDE policy MLP φ_ω."""

    policy_activation: str = "elu"
    """Activation function for policy network.  ELU avoids dead neurons."""

    use_residual: bool = True
    """If True, use residual connections in the policy network."""

    # --- Transaction Costs ---
    transaction_cost_bps: float = 15.0
    """c: Proportional transaction cost in basis points (15 bps = 0.0015).
    Represents exchange fees + average slippage."""

    huber_delta: float = 0.01
    """δ: Huber transition threshold (1% turnover).
    Below δ: quadratic penalty (smooth gradient near zero).
    Above δ: linear penalty (realistic exchange fees)."""

    # --- Utility ---
    risk_aversion: float = 2.0
    """γ: Coefficient of relative risk aversion for CRRA utility.
    U(X) = X^{1−γ} / (1−γ).
    γ → 1: Kelly criterion (log utility).
    γ > 1: more conservative (penalises extreme losses)."""

    # --- Wealth Process ---
    initial_wealth: float = 1.0
    """X_0: Initial portfolio wealth (normalised to 1.0)."""

    risk_free_rate: float = 0.02
    """r_f: Continuous annual risk-free rate (same as Phase 3)."""

    # --- Constraints ---
    allow_short_selling: bool = False
    """If False, W_t ≥ 0 (long-only, probability simplex).
    If True, W_t ∈ ℝ^N with Σ W_i = 1 (allows negative weights)."""

    max_leverage: float = 1.0
    """Maximum total portfolio leverage: Σ |W_i| ≤ max_leverage."""

    n_monte_carlo_paths: int = 1024
    """M: Number of synthetic paths from Phase 3 used for
    Monte Carlo expectation in the outer loop."""


# ---------------------------------------------------------------------------
#  Training: Bi-Level TTSA Optimisation (Ch. 6)
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Configuration for the bi-level Stackelberg training loop.

    Reference: Proposal Ch. 6 — Bi-level Optimization & Training Dynamics.

    Inner loop (§6.2, fast-scale):
        L_inner = L_DSM + η₁·L_MSB + η₂·L_ev + η₃·L_SGW
        θ_{k+1} = θ_k − α_k ∇_θ L_inner(θ_k)

    Outer loop (§6.3, slow-scale):
        L_outer = −E[U(X_T)] + friction
        ω_{k+1} = ω_k − β_k ∇_ω L_outer(ω_k)

    Robbins-Monro conditions (§6.4):
        Σ α_k = ∞,  Σ α_k² < ∞,  lim β_k/α_k = 0
    """

    # --- Epochs & Steps ---
    n_epochs: int = 200
    """Total number of training epochs."""

    n_inner_steps_per_outer: int = 5
    """Number of inner (generator) gradient steps per single outer
    (controller) gradient step.  Enforces the timescale separation."""

    batch_size: int = 32
    """Mini-batch size for the data loader."""

    # --- Inner Loop (Fast-Scale) Learning Rate ---
    inner_lr_initial: float = 1e-3
    """α_0: Initial learning rate for the generative block θ."""

    inner_lr_decay_power: float = 0.6
    """Polynomial decay exponent for α_k = α_0 / k^p.
    Must satisfy: 0.5 < p < 1 for Robbins-Monro convergence."""

    inner_optimizer: str = "adamw"
    """Optimiser for the inner (generative) loop."""

    inner_weight_decay: float = 1e-5
    """Weight decay (L2 regularisation) for inner optimiser."""

    # --- Outer Loop (Slow-Scale) Learning Rate ---
    outer_lr_initial: float = 1e-4
    """β_0: Initial learning rate for the control block ω.
    Must satisfy β_0 << α_0 for timescale separation."""

    outer_lr_decay_power: float = 0.9
    """Polynomial decay exponent for β_k = β_0 / k^q.
    Must satisfy q > p to guarantee lim β_k/α_k = 0."""

    outer_optimizer: str = "adamw"
    """Optimiser for the outer (controller) loop."""

    outer_weight_decay: float = 1e-4
    """Weight decay for outer optimiser."""

    # --- Inner Loss Coefficients ---
    eta_msb: float = 10.0
    """η₁: Weight for the Martingale Schrödinger Bridge penalty L_MSB."""

    eta_evidential: float = 1.0
    """η₂: Weight for the evidential Dirichlet loss L_ev."""

    eta_sgw: float = 0.1
    """η₃: Weight for the SGW alignment loss L_SGW."""

    lambda_kl_evidential: float = 0.01
    """λ_KL: KL divergence regularisation coefficient in L_ev.
    Penalises Dirichlet divergence from uniform prior for OOD data."""

    # --- Mixed Precision (§8.2.2) ---
    use_amp: bool = True
    """Enable Automatic Mixed Precision (BF16/FP16) training."""

    amp_dtype: str = "bfloat16"
    """AMP dtype: 'bfloat16' preserves exponent range (better for
    matrix operations), 'float16' for maximum throughput."""

    # --- Gradient Clipping ---
    max_grad_norm: float = 1.0
    """Maximum gradient norm for gradient clipping (prevents explosion)."""

    # --- Checkpointing ---
    checkpoint_every_n_epochs: int = 10
    """Save model checkpoint every N epochs."""

    checkpoint_dir: str = "checkpoints"
    """Directory for saving model checkpoints."""

    early_stopping_patience: int = 20
    """Stop training if validation loss doesn't improve for N epochs."""


# ---------------------------------------------------------------------------
#  Evaluation Protocols (Ch. 7)
# ---------------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    """Configuration for the evaluation & benchmarking suite.

    Reference: Proposal Ch. 7 — Evaluation Protocols & Benchmarking.

    Metrics:
        1. AVR (§7.1): Arbitrage Violation Rate
        2. SFPS (§7.2): Stylized Facts Preservation Score (KS test)
        3. NSR (§7.3): Net Sharpe Ratio under Huber friction
        4. GOP (§7.3.2): Growth Optimal Portfolio trajectory
    """

    # --- Arbitrage Violation Rate (§7.1) ---
    avr_tolerance: float = 1e-5
    """ε_tol: Numerical tolerance for AVR computation.
    Relative discrepancies below this are attributed to floating-point
    arithmetic, not genuine arbitrage hallucinations."""

    # --- Stylized Facts (§7.2) ---
    sfps_weights: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])
    """w_k: Weights for [heavy_tails, vol_clustering, leverage_effect].
    Must sum to 1.0.  Higher weight on tails and vol clustering as they
    are most critical for systemic risk assessment."""

    sfps_max_acf_lag: int = 100
    """Maximum lag for autocorrelation function in volatility clustering."""

    sfps_leverage_max_lag: int = 25
    """Maximum lag for leverage effect computation."""

    # --- Net Sharpe Ratio (§7.3) ---
    annualization_factor: int = 252
    """A: Annualisation factor.  252 for daily, 252×390 for minute-level."""

    risk_free_rate_annual: float = 0.02
    """Annualised risk-free rate for NSR excess return computation."""

    # --- Plotting ---
    plot_format: str = "pdf"
    """Output format for academic plots: 'pdf' or 'svg' (vector)."""

    plot_dpi: int = 300
    """DPI for rasterised elements within vector plots."""

    plot_font_family: str = "Times New Roman"
    """Font family for publication-quality figures."""

    plot_font_size: int = 12
    """Base font size for plots."""


# ---------------------------------------------------------------------------
#  Data Pipeline (§7.4)
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Configuration for data ingestion and preprocessing.

    Reference: Proposal §7.4 — Dataset Architecture: High-Frequency LOB
    Topography and Feature Calibration.
    """

    # --- LOB Structure ---
    n_assets: int = 12
    """N: Number of assets in the cross-sectional universe.
    12 for initial development (Fama-French 12-industry).
    500 for full-scale deployment."""

    n_features: int = 3
    """F: Number of engineered features per asset.
    [micro_price, OFI, liquidity_skew]."""

    lob_depth: int = 10
    """D: Number of price levels on each side of the LOB."""

    # --- Causal Normalisation (§7.4.3) ---
    normalisation_window: int = 120
    """W: Rolling window size for causal Z-score normalisation.
    Default: 120 ticks (~2 hours of HFT data)."""

    # --- Spline Interpolation ---
    spline_order: int = 3
    """Order of the natural cubic spline (3 = cubic)."""

    # --- Train / Val / Test Split ---
    train_ratio: float = 0.7
    """Fraction of data for training (chronological split)."""

    val_ratio: float = 0.15
    """Fraction of data for validation."""

    test_ratio: float = 0.15
    """Fraction of data for out-of-sample testing."""

    # --- Sequence Parameters ---
    lookback_window: int = 60
    """Number of historical time-steps per input sample."""

    forecast_horizon: int = 10
    """Number of future time-steps to generate / forecast."""

    # --- Liquidity Skew Decay (§7.4.2) ---
    liquidity_skew_kappa: float = 0.5
    """κ: Exponential decay scalar for multi-level liquidity skew.
    S_skew(t) = Σ_d exp(−κ·d) · (V^b_d − V^a_d)/(V^b_d + V^a_d)."""

    # --- Data Paths ---
    data_dir: str = "data/raw"
    """Directory containing raw LOB / market data files."""

    processed_dir: str = "data/processed"
    """Directory for processed / cached feature tensors."""


# ---------------------------------------------------------------------------
#  Master Configuration
# ---------------------------------------------------------------------------
@dataclass
class MasterConfig:
    """Top-level configuration aggregating all sub-configs.

    This is the single source of truth passed to every module.
    Instantiate with defaults:
        config = MasterConfig()
    or override specific fields:
        config = MasterConfig(phase0=Phase0Config(hidden_dim=512))
    """

    phase0: Phase0Config = field(default_factory=Phase0Config)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    sgw: SGWConfig = field(default_factory=SGWConfig)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # --- Global ---
    seed: int = 42
    """Global random seed for reproducibility."""

    device: str = "cuda"
    """Target device: 'cuda', 'cpu', or 'cuda:0', etc."""

    project_name: str = "quantum_topo_diffusion"
    """Project name for logging and checkpointing."""

    @property
    def n_assets(self) -> int:
        """Convenience accessor for the asset universe size."""
        return self.data.n_assets

    @property
    def input_dim(self) -> int:
        """N × F: Total flattened input dimension for the Neural CDE."""
        return self.data.n_assets * self.data.n_features

    @property
    def tangent_dim(self) -> int:
        """N(N+1)/2: Dimension of the flattened tangent vector from Phase 2."""
        n = self.data.n_assets
        return n * (n + 1) // 2
