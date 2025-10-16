//! Configuration types shared across estimators.

#[derive(Copy, Clone, Debug)]
pub(crate) struct WosParams {
    /// Distance to boundary below which the walk terminates.
    pub epsilon: f32,
    /// Hard cap on the number of steps.
    pub max_steps: u32,
}

/// Budget for a walk, intended for higher-level solver orchestration.
#[derive(Copy, Clone, Debug)]
pub struct WalkBudget {
    /// Distance to boundary below which the walk terminates.
    pub epsilon: f32,
    /// Hard cap on the number of steps.
    pub max_steps: u32,
}

impl WalkBudget {
    pub const fn new(epsilon: f32, max_steps: u32) -> Self {
        Self { epsilon, max_steps }
    }

    #[inline]
    pub(crate) fn to_wos(self) -> WosParams {
        WosParams {
            epsilon: self.epsilon,
            max_steps: self.max_steps,
        }
    }
}

impl From<WosParams> for WalkBudget {
    fn from(params: WosParams) -> Self {
        Self {
            epsilon: params.epsilon,
            max_steps: params.max_steps,
        }
    }
}

impl From<WalkBudget> for WosParams {
    fn from(value: WalkBudget) -> Self {
        value.to_wos()
    }
}

/// How to sample interior points Y inside each WoS ball.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum InteriorSampling {
    /// Uniform in ball: Y ~ Uniform(B(x,R))
    Uniform,
    /// Green-ball importance sampling: pdf p(y) ∝ G_B^{3D}(x,y)
    ///
    /// With this choice the integral ∫_B G f dy becomes Z * E[f(Y)]
    /// where Z = ∫_B G dy = R²/6, so the weight is constant per step.
    GreenBall,
}

/// Unbiased estimator for the Dirichlet Poisson problem **in 3D** using WoS + per-step
/// Monte Carlo integration inside each largest empty ball.
#[derive(Copy, Clone, Debug)]
pub struct PoissonParams {
    /// Number of interior samples per WoS step. `1` is unbiased and cheapest;
    /// larger values reduce variance.
    pub interior_samples_per_step: u32,
    /// Clamp for near-singularity in Green_B at r≈0 (numerical safety).
    pub min_r: f32,
    pub sampling: InteriorSampling,
}

impl PoissonParams {
    pub const fn new(interior_samples_per_step: u32) -> Self {
        Self {
            interior_samples_per_step,
            min_r: 1e-7,
            sampling: InteriorSampling::Uniform,
        }
    }
    pub const fn with_min_r(self, min_r: f32) -> Self {
        Self { min_r, ..self }
    }
    pub const fn with_sampling(self, sampling: InteriorSampling) -> Self {
        Self { sampling, ..self }
    }
}

/// Gradient estimator knobs (used by both Laplace/Poisson).
#[derive(Copy, Clone, Debug)]
pub struct GradParams {
    /// Number of sphere directions ξ per gradient call (surface term).
    pub boundary_dirs: u32,
    /// Number of interior samples for the Poisson volume term.
    pub interior_samples: u32,
    /// How to sample the Poisson volume term.
    pub sampling: InteriorSampling,
    /// Clamp for near-singularity in kernels.
    pub min_r: f32,
}

impl GradParams {
    pub fn new(boundary_dirs: u32, interior_samples: u32) -> Self {
        Self {
            boundary_dirs: boundary_dirs.max(1),
            interior_samples: interior_samples.max(1),
            sampling: InteriorSampling::Uniform,
            min_r: 1e-7,
        }
    }
    pub const fn with_sampling(self, sampling: InteriorSampling) -> Self {
        Self { sampling, ..self }
    }
    pub const fn with_min_r(self, min_r: f32) -> Self {
        Self { min_r, ..self }
    }
}
