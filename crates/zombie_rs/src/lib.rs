#![no_std]

//! Grid-free Monte Carlo geometry processing kernel.
//!
//! This crate provides safe, `no_std` implementations of Walk-on-Spheres based
//! estimators for Laplace and Poisson problems on volumetric domains, along with
//! domain abstractions, acceleration structures, and utilities for future
//! extensions such as Walk-on-Stars and Robin boundary conditions.

extern crate alloc;

pub mod accel;
pub mod boundary;
pub mod domain;
pub mod estimators;
pub mod math;
pub mod observer;
pub mod params;
pub mod rng;
pub mod sampling;
pub mod solver;
pub mod source;
pub mod stats;

pub use accel::{Bvh, BvhAccel, ClosestAccel, ClosestNaive};
pub use boundary::{BoundaryDirichlet, BoundaryDirichletFn};
pub use domain::sdf_csg;
pub use domain::{Closest, Domain, PolygonSoupDomain, SdfDomain};
pub use estimators::{
    grad_laplace_dirichlet_wos, grad_poisson_dirichlet_wos, wos_laplace_dirichlet,
    wos_poisson_dirichlet,
};
pub use math::{Aabb, Vec3, closest_point_on_triangle};
pub use observer::{
    NoopObserver, PlyRecorder, StatsObserver, TerminationReason, WalkObserver, WalkOutcome,
    WalkStatsSnapshot,
};
pub use params::{GradParams, InteriorSampling, PoissonParams, WalkBudget};
pub use rng::Rng;
pub use solver::{Solver, SolverBuilder};
pub use source::{PointSource, SourceTerm};
pub use stats::Stats;

#[cfg(test)]
mod tests {
    use crate::*;

    /// Run many **Poisson** single-path estimates of u(x) and return Welford stats.
    fn poisson_repeat_and_stats<'a, D, A, G, F>(
        solver: &Solver<'a, D, A>,
        trials: u32,
        g: &G,
        fsrc: &F,
        walk: WalkBudget,
        poisson: PoissonParams,
        x: Vec3,
        seed: u64,
    ) -> Stats
    where
        D: Domain,
        A: ClosestAccel<D>,
        G: BoundaryDirichlet,
        F: SourceTerm,
    {
        let mut stats = Stats::default();
        let mut rng = rng::Rng::seed_from(seed);
        for _ in 0..trials {
            let u = solver.poisson_dirichlet(g, fsrc, walk, poisson, &mut rng, x);
            stats.push(u);
        }
        stats
    }

    #[test]
    fn laplacian_dirichlet_smoke() {
        // Define a sphere SDF of radius 1 at origin:
        let sphere = SdfDomain::new(|p: Vec3| p.length() - 1.0);
        let accel = ClosestNaive;
        let solver = Solver::builder(&sphere, &accel).build();

        // Dirichlet boundary: u = 1 on boundary
        let bc = BoundaryDirichletFn::new(|_p| 1.0);

        let budget = WalkBudget::new(1e-4, 10_000);
        let mut rng = rng::Rng::seed_from(42);

        // Evaluate near center (harmonic extension is constant 1.0 for this bc)
        let u = solver.laplace_dirichlet(&bc, budget, &mut rng, Vec3::new(0.0, 0.0, 0.0));
        assert!((u - 1.0).abs() < 5e-2); // stochastic tolerance
    }

    /// Constant Dirichlet Poisson in the unit ball:
    ///  - domain: φ(p)=|p|-1 (unit ball),
    ///  - PDE:    -Δ u = 1 in Ω,  u = 0 on ∂Ω,
    ///  - exact:  u(r) = (1 - r^2)/6  in 3D.
    ///
    /// We test at the center (r=0): u(0) = 1/6 ≈ 0.166666...
    #[test]
    fn poisson_unit_ball_constant_source_center() {
        let sphere = SdfDomain::new(|p: Vec3| p.length() - 1.0);
        let accel = ClosestNaive;
        let solver = Solver::builder(&sphere, &accel).build();
        let g0 = BoundaryDirichletFn::new(|_p| 0.0);
        // Source f ≡ 1  (matches -Δ u = 1)
        struct One;
        impl SourceTerm for One {
            #[inline]
            fn value(&self, _x: Vec3) -> f32 {
                1.0
            }
        }
        let fsrc = One;

        let walk = WalkBudget::new(1e-4, 20_000);
        let poisson = PoissonParams::new(8); // 8 interior samples per step helps variance
        let mut rng = rng::Rng::seed_from(1234);

        // Expect ~1/6 at center
        let u = solver.poisson_dirichlet(
            &g0,
            &fsrc,
            walk,
            poisson,
            &mut rng,
            Vec3::new(0.0, 0.0, 0.0),
        );

        // Loose MC tolerance; tighten as needed.
        let exact = 1.0 / 6.0;
        assert!((u - exact).abs() < 0.06, "u(0)≈{exact}, got {u}");
    }

    /// Regression: when f ≡ 0, Poisson reduces to Laplace estimator.
    #[test]
    fn poisson_reduces_to_laplace_when_source_zero() {
        let sphere = SdfDomain::new(|p: Vec3| p.length() - 1.0);
        let accel = ClosestNaive;
        let solver = Solver::builder(&sphere, &accel).build();
        let g = BoundaryDirichletFn::new(|_p| 1.0);
        struct Zero;
        impl SourceTerm for Zero {
            #[inline]
            fn value(&self, _x: Vec3) -> f32 {
                0.0
            }
        }

        let walk = WalkBudget::new(1e-4, 10_000);
        let poisson = PoissonParams::new(1);
        let mut rng1 = rng::Rng::seed_from(7);
        let mut rng2 = rng::Rng::seed_from(7);

        let u_lap = solver.laplace_dirichlet(&g, walk, &mut rng1, Vec3::new(0.0, 0.0, 0.0));
        let u_pois = solver.poisson_dirichlet(
            &g,
            &Zero,
            walk,
            poisson,
            &mut rng2,
            Vec3::new(0.0, 0.0, 0.0),
        );

        assert!(
            (u_lap - u_pois).abs() < 5e-2,
            "Poisson(f=0) should match Laplace"
        );
    }

    /// Variance reduction: Green-ball IS vs Uniform for constant source.
    /// Setup:
    ///   -Ω = unit ball, g=0,  -Δu = 1  =>  u(r) = (1 - r^2)/6,  u(0)=1/6.
    /// Green-ball IS turns each ball's interior estimate into Z * mean(f)=Z exactly per step
    /// (for f≡1), so it removes interior sampling noise; only path randomness remains.
    #[test]
    fn variance_reduction_greenball_vs_uniform() {
        let domain = SdfDomain::new(|p: Vec3| p.length() - 1.0);
        let accel = ClosestNaive;
        let solver = Solver::builder(&domain, &accel).build();
        let g0 = BoundaryDirichletFn::new(|_p| 0.0);

        struct One;
        impl SourceTerm for One {
            fn value(&self, _x: Vec3) -> f32 {
                1.0
            }
        }
        let fsrc = One;

        let walk = WalkBudget::new(1e-4, 50_000);
        let x0 = Vec3::new(0.2, 0.0, 0.0);

        // Keep interior_samples_per_step small to highlight per-step variance effects
        let uni = PoissonParams::new(1).with_sampling(InteriorSampling::Uniform);
        let grn = PoissonParams::new(1).with_sampling(InteriorSampling::GreenBall);

        let trials = 400; // modest but illustrative
        let s_uni = poisson_repeat_and_stats(&solver, trials, &g0, &fsrc, walk, uni, x0, 777);
        let s_grn = poisson_repeat_and_stats(&solver, trials, &g0, &fsrc, walk, grn, x0, 888);

        // Both unbiased; check means near truth
        let exact = (1.0 - x0.length_sq()) / 6.0;
        assert!(
            (s_uni.mean() - exact).abs() < 0.08,
            "uniform mean off: {} vs {}",
            s_uni.mean(),
            exact
        );
        assert!(
            (s_grn.mean() - exact).abs() < 0.08,
            "green mean off: {} vs {}",
            s_grn.mean(),
            exact
        );

        // Variance reduction: should be strictly lower with Green-ball IS
        // Leave some slack: stochastic—just ensure it's noticeably smaller.
        assert!(
            s_grn.var() < s_uni.var() * 0.8,
            "expected variance drop; got uni={}, green={}",
            s_uni.var(),
            s_grn.var()
        );
    }

    #[test]
    fn point_source_analytic_kick_is_unbiased_in_unit_ball() {
        use crate::*;

        // Domain: unit ball via SDF
        let ball = SdfDomain::new(|p: Vec3| p.length() - 1.0);
        let accel = ClosestNaive;
        let solver = Solver::builder(&ball, &accel).build();

        // Dirichlet: g ≡ 0
        let g0 = BoundaryDirichletFn::new(|_| 0.0);

        // Point source at z (not at x to avoid min_r clamp dominating); strength q
        let z = Vec3::new(0.25, 0.0, 0.0);
        let q = 1.0;
        let fsrc = PointSource {
            position: z,
            strength: q,
        };

        // WoS configuration
        let walk = WalkBudget::new(1e-4, 10_000);
        let poisson = PoissonParams::new(1);
        let mut rng = rng::Rng::seed_from(123);

        // Evaluate at x = 0
        let x = Vec3::new(0.0, 0.0, 0.0);
        let u = solver.poisson_dirichlet(&g0, &fsrc, walk, poisson, &mut rng, x);

        // Exact Green of unit ball at center:
        // G(0,z) = (1/(4π)) (1/|z| - 1)
        let r = (z - x).length().max(1e-7);
        let exact = (1.0 / (4.0 * core::f32::consts::PI)) * (1.0 / r - 1.0) * q;

        assert!((u - exact).abs() < 1e-4, "u = {u}, exact = {exact}");
    }

    #[test]
    fn grad_laplace_dirichlet_linear_bc_is_constant() {
        let ball = SdfDomain::new(|p: Vec3| p.length() - 1.0);
        let accel = ClosestNaive;
        let solver = Solver::builder(&ball, &accel).build();
        // g(p)=p.x ⇒ u(x)=x.x inside the ball (harmonic extension), ∇u=(1,0,0).
        let g_lin = BoundaryDirichletFn::new(|p: Vec3| p.x);
        let walk = WalkBudget::new(1e-4, 40_000);
        let mut rng = rng::Rng::seed_from(2025);

        // interior point
        let x = Vec3::new(0.2, -0.1, 0.15);

        // A few dozen directions is enough; increase for tighter tolerance.
        let grad_cfg = GradParams::new(512, 1);

        let g_est = solver.laplace_gradient(&g_lin, walk, grad_cfg, &mut rng, x);
        let g_true = Vec3::new(1.0, 0.0, 0.0);
        let err = (g_est - g_true).length();

        assert!(
            err < 0.1,
            "Laplace grad error too large: got {g_est:?}, want {g_true:?}, |err|={err}"
        );
    }

    #[test]
    fn grad_poisson_constant_source_matches_analytic() {
        let ball = SdfDomain::new(|p: Vec3| p.length() - 1.0);
        let accel = ClosestNaive;
        let solver = Solver::builder(&ball, &accel).build();
        let g0 = BoundaryDirichletFn::new(|_| 0.0);
        struct One;
        impl SourceTerm for One {
            fn value(&self, _x: Vec3) -> f32 {
                1.0
            }
        }
        let fsrc = One;

        let walk = WalkBudget::new(1e-4, 60_000);
        // Keep Poisson scalar estimator variance low (used for surface term evaluation):
        let pois = PoissonParams::new(4).with_sampling(InteriorSampling::GreenBall);
        let mut rng = rng::Rng::seed_from(7);

        // interior point
        let x = Vec3::new(0.25, -0.1, 0.2);

        // More samples for the gradient’s surface+volume terms
        let grad_cfg = GradParams::new(256, 64).with_sampling(InteriorSampling::GreenBall);

        let g_est = solver.poisson_gradient(&g0, &fsrc, walk, pois, grad_cfg, &mut rng, x);

        // Analytic: u=(1 - |x|^2)/6 ⇒ ∇u = -(1/3) x
        let g_true = x * (-1.0 / 3.0);
        let err = (g_est - g_true).length();

        assert!(
            err < 0.12,
            "Poisson grad error too large: got {g_est:?}, want {g_true:?}, |err|={err}"
        );
    }

    #[test]
    fn observers_capture_walk_data() {
        let sphere = SdfDomain::new(|p: Vec3| p.length() - 1.0);
        let accel = ClosestNaive;
        let stats = StatsObserver::new();
        let ply = PlyRecorder::new();
        let solver = Solver::builder(&sphere, &accel)
            .with_observer(stats.clone())
            .with_observer(ply.clone())
            .build();

        let bc = BoundaryDirichletFn::new(|_| 0.0);
        let mut rng = rng::Rng::seed_from(5);
        let _ = solver.laplace_dirichlet(
            &bc,
            WalkBudget::new(1e-4, 2_000),
            &mut rng,
            Vec3::new(0.1, 0.1, 0.1),
        );

        let snap = stats.snapshot();
        assert_eq!(snap.walks, 1);
        assert!(snap.total_steps > 0);

        let ply_text = ply.to_ascii();
        assert!(ply_text.contains("ply"));
        assert!(ply_text.contains("vertex"));
    }
}
