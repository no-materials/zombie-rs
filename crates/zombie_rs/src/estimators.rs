//! Core Monte Carlo estimators.
//!
//! The routines in this module implement Walk-on-Spheres estimators for Laplace
//! and Poisson equations, along with gradient estimators. Each solver reports a
//! [`WalkOutcome`](crate::observer::WalkOutcome) that surfaces the termination
//! reason so that higher-level orchestration (e.g. diagnostics) can react safely.

use core::f32::consts::PI;

use crate::accel::ClosestAccel;
use crate::boundary::BoundaryDirichlet;
use crate::domain::Domain;
use crate::math::Vec3;
use crate::observer::{
    TerminationReason, WalkObserver, WalkOutcome, WalkStart, WalkStep, WalkTerminate,
};
use crate::params::{GradParams, InteriorSampling, PoissonParams, WalkBudget, WosParams};
use crate::rng::Rng;
use crate::sampling::{
    ball_volume, green_ball_3d, green_ball_total_mass, sample_ball_by_green_pdf,
    sample_ball_uniform, wos_jump,
};
use crate::source::SourceTerm;

/// Walk-on-Spheres Laplace estimator returning both the value and termination metadata.
pub fn wos_laplace_dirichlet<D, G, A, O>(
    domain: &D,
    accel: &A,
    g: &G,
    budget: WalkBudget,
    rng: &mut Rng,
    mut x: Vec3,
    observer: &O,
) -> WalkOutcome
where
    D: Domain,
    A: ClosestAccel<D>,
    G: BoundaryDirichlet,
    O: WalkObserver,
{
    let params: WosParams = budget.into();
    let mut steps = 0u32;
    debug_assert!(
        domain.is_inside(x),
        "wos_laplace_dirichlet: x must start inside Ω"
    );

    observer.on_start(WalkStart { position: x });

    loop {
        let c = accel.closest(domain, x);
        if c.distance <= params.epsilon {
            observer.on_terminate(WalkTerminate {
                position: c.point,
                reason: TerminationReason::HitBoundary,
                depth: steps,
            });
            return WalkOutcome::new(g.value(c.point), TerminationReason::HitBoundary, steps);
        }

        observer.on_step(WalkStep {
            position: x,
            radius: c.distance,
            depth: steps,
        });
        x = wos_jump(x, c.distance, rng);
        steps += 1;
        if steps >= params.max_steps {
            let c2 = accel.closest(domain, x);
            observer.on_terminate(WalkTerminate {
                position: c2.point,
                reason: TerminationReason::MaxSteps,
                depth: steps,
            });
            return WalkOutcome::new(g.value(c2.point), TerminationReason::MaxSteps, steps);
        }
    }
}

/// Poisson (Dirichlet) estimator via Walk-on-Spheres + ball Green function, including metadata.
pub fn wos_poisson_dirichlet<D, A, G, F, O>(
    domain: &D,
    accel: &A,
    g: &G,
    fsrc: &F,
    budget: WalkBudget,
    poisson: PoissonParams,
    rng: &mut Rng,
    mut x: Vec3,
    observer: &O,
) -> WalkOutcome
where
    D: Domain,
    A: ClosestAccel<D>,
    G: BoundaryDirichlet,
    F: SourceTerm,
    O: WalkObserver,
{
    debug_assert!(
        domain.is_inside(x),
        "wos_poisson_dirichlet: x must be inside Ω"
    );

    let params: WosParams = budget.into();
    let mut steps = 0u32;
    let mut acc = 0.0_f32;

    // Optional point source
    let point = fsrc.point_params();

    observer.on_start(WalkStart { position: x });

    loop {
        let c = accel.closest(domain, x);
        let radius = c.distance;

        if radius <= params.epsilon {
            observer.on_terminate(WalkTerminate {
                position: c.point,
                reason: TerminationReason::HitBoundary,
                depth: steps,
            });
            return WalkOutcome::new(
                acc + g.value(c.point),
                TerminationReason::HitBoundary,
                steps,
            );
        }

        observer.on_step(WalkStep {
            position: x,
            radius,
            depth: steps,
        });

        // -------- Point-source analytic contribution (if present)
        if let Some((z, q)) = point {
            let r = (z - x).length();
            if r <= radius {
                // The δ integrates exactly over the ball: ∫_B G δ = G_B(x,z)
                acc += q * green_ball_3d(radius, r.max(poisson.min_r));
                // Note: keep walking; other balls may contribute nothing for this δ.
            }
        }

        // -------- Volume term for general sources
        let m = poisson.interior_samples_per_step.max(1);
        match poisson.sampling {
            InteriorSampling::Uniform => {
                // I ≈ Vol(B) * (1/m) Σ G_B(x,Y) f(Y),  Y ~ Uniform(B)
                let vol = ball_volume(radius);
                let inv_m = 1.0 / (m as f32);
                let mut sum = 0.0;
                for _ in 0..m {
                    let y = sample_ball_uniform(rng, x, radius);
                    let r = (y - x).length().max(poisson.min_r);
                    let green_value = green_ball_3d(radius, r);
                    // For point sources, f.value(y)=0 almost surely, which is fine;
                    // the analytic kick above already handled the δ contribution.
                    sum += green_value * fsrc.value(y);
                }
                acc += vol * (sum * inv_m);
            }
            InteriorSampling::GreenBall => {
                // With p(y) ∝ G_B, the integral becomes Z * E[f(Y)] with Z = R^2/6.
                // This eliminates the G_B factor from the MC sum (nice variance cut).
                let z_mass = green_ball_total_mass(radius);
                let inv_m = 1.0 / (m as f32);
                let mut sum = 0.0;
                for _ in 0..m {
                    let y = sample_ball_by_green_pdf(rng, x, radius);
                    sum += fsrc.value(y);
                }
                acc += z_mass * (sum * inv_m);
            }
        }

        // Step
        x = wos_jump(x, radius, rng);

        steps += 1;
        if steps >= params.max_steps {
            let c2 = accel.closest(domain, x);
            observer.on_terminate(WalkTerminate {
                position: c2.point,
                reason: TerminationReason::MaxSteps,
                depth: steps,
            });
            return WalkOutcome::new(acc + g.value(c2.point), TerminationReason::MaxSteps, steps);
        }
    }
}

/// Generic single-ball surface-term gradient estimator:
/// returns (3/R) * E[ U(x+R ξ) * ξ ], where U is "how to evaluate u(·)".
fn grad_surface_single_ball<D, A, O, Ueval>(
    domain: &D,
    accel: &A,
    eval_u: &mut Ueval,
    observer: &O,
    walk: WosParams,
    boundary_dirs: u32,
    min_r: f32,
    rng: &mut Rng,
    x: Vec3,
) -> Vec3
where
    D: Domain,
    A: ClosestAccel<D>,
    O: WalkObserver,
    Ueval: FnMut(&D, &A, WosParams, &O, &mut Rng, Vec3) -> f32,
{
    let c = accel.closest(domain, x);
    let radius = (c.distance).max(min_r);
    let m = boundary_dirs.max(1);
    let mut acc = Vec3::new(0.0, 0.0, 0.0);
    for _ in 0..m {
        let xi = crate::sampling::sample_unit_sphere(rng);
        let xp = x + xi * radius;
        let up = eval_u(domain, accel, walk, observer, rng, xp);
        acc += xi * up;
    }
    acc * (3.0 / (radius * m as f32))
}

/// Gradient of the Laplace–Dirichlet solution via a single-ball estimator.
///
/// Observer callbacks are invoked for the continuation walks sampled during the estimate.
pub fn grad_laplace_dirichlet_wos<D, A, G, O>(
    domain: &D,
    accel: &A,
    g: &G,
    walk: WalkBudget,
    observer: &O,
    grad: GradParams,
    rng: &mut Rng,
    x: Vec3,
) -> Vec3
where
    D: Domain,
    A: ClosestAccel<D>,
    G: BoundaryDirichlet,
    O: WalkObserver,
{
    let walk_params: WosParams = walk.into();
    let mut eval_u = |d: &D, a: &A, w: WosParams, obs: &O, r: &mut Rng, xp: Vec3| {
        let budget = WalkBudget::from(w);
        wos_laplace_dirichlet(d, a, g, budget, r, xp, obs).value
    };
    grad_surface_single_ball(
        domain,
        accel,
        &mut eval_u,
        observer,
        walk_params,
        grad.boundary_dirs,
        grad.min_r,
        rng,
        x,
    )
}

/// Gradient of the Poisson–Dirichlet solution in 3D via a single-ball estimator.
///
/// Observer callbacks are invoked for the continuation walks sampled during the estimate.
pub fn grad_poisson_dirichlet_wos<D, A, G, F, O>(
    domain: &D,
    accel: &A,
    g: &G,
    fsrc: &F,
    walk: WalkBudget,
    observer: &O,
    pois: PoissonParams,
    grad: GradParams,
    rng: &mut Rng,
    x: Vec3,
) -> Vec3
where
    D: Domain,
    A: ClosestAccel<D>,
    G: BoundaryDirichlet,
    F: SourceTerm,
    O: WalkObserver,
{
    // Ball at x
    let c = accel.closest(domain, x);
    let radius = c.distance.max(grad.min_r);

    // Surface term (needs full Poisson u)
    let walk_params: WosParams = walk.into();
    let mut eval_u = |d: &D, a: &A, w: WosParams, obs: &O, r: &mut Rng, xp: Vec3| {
        let budget = WalkBudget::from(w);
        wos_poisson_dirichlet(d, a, g, fsrc, budget, pois, r, xp, obs).value
    };
    let surf = grad_surface_single_ball(
        domain,
        accel,
        &mut eval_u,
        observer,
        walk_params,
        grad.boundary_dirs,
        grad.min_r,
        rng,
        x,
    );

    // --- Volume term:  ∫_B ∇_x G f
    let k = grad.interior_samples.max(1);
    let vol = match grad.sampling {
        InteriorSampling::Uniform => {
            // I ≈ Vol(B) * (1/k) Σ [ ∇_x G(x, Y_i) f(Y_i) ], Y_i ~ Unif(B)
            let vol_ball = ball_volume(radius);
            let mut sum = Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..k {
                let y = sample_ball_uniform(rng, x, radius);
                let r = (x - y).length().max(grad.min_r);
                let grad_g = (x - y) * (-1.0 / (4.0 * PI * r * r * r)); // -(x-y)/4π r^3
                sum += grad_g * fsrc.value(y);
            }
            sum * (vol_ball / k as f32)
        }
        InteriorSampling::GreenBall => {
            // p(y) ∝ G(x,y) (with total mass Z = R²/6)
            // ∫ ∇G f = E_p[ (∇G / p) f ] = Z * E_p[ (∇G / G) f ]
            let z_mass = green_ball_total_mass(radius);
            let mut sum = Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..k {
                let y = sample_ball_by_green_pdf(rng, x, radius);
                let r = (x - y).length().max(grad.min_r);
                let gval = (1.0 / (4.0 * PI)) * (1.0 / r - 1.0 / radius).max(1e-12);
                let grad_g = (x - y) * (-1.0 / (4.0 * PI * r * r * r));
                // weight = Z * (gradG / G)
                let w = z_mass / gval;
                sum += grad_g * (w * fsrc.value(y));
            }
            sum * (1.0 / k as f32)
        }
    };

    // ∇u = surface − volume
    surf - vol
}
