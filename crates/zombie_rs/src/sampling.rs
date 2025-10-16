//! Random sampling utilities for spheres and balls.

use core::f32::consts::PI;
use libm::{cosf, powf, sinf, sqrtf};

use crate::math::Vec3;
use crate::rng::Rng;

/// Uniform sampling on the unit sphere (S²)
///
/// Uses a branch-free method based on two uniforms.
#[inline]
pub(crate) fn sample_unit_sphere(rng: &mut Rng) -> Vec3 {
    // Marsaglia method
    let u = 2.0 * rng.uniform_f32() - 1.0; // z in [-1,1]
    let t = 2.0 * PI * rng.uniform_f32();
    let r = sqrtf(1.0 - u * u);
    Vec3::new(r * cosf(t), r * sinf(t), u)
}

/// Draw Y ∈ B(center,R) uniformly (volume pdf).
#[inline]
pub(crate) fn sample_ball_uniform(rng: &mut Rng, center: Vec3, radius: f32) -> Vec3 {
    let dir = sample_unit_sphere(rng); // S²
    let u = rng.uniform_f32().max(1e-7); // radius ~ R * u^(1/3)
    center + dir * (radius * powf(u, 1.0 / 3.0))
}

/// Inverse CDF for t = r/R under the 3D “Green-ball” radius pdf.
/// F(t) = 3 t^2 - 2 t^3,  t ∈ (0,1).
fn inv_cdf_radius_under_green_pdf(u: f32) -> f32 {
    // Clamp u to (0,1) and do a few Newton steps; 3 steps are plenty.
    let mut t = u.min(1.0 - 1e-7).max(1e-7);
    for _ in 0..3 {
        let f = 3.0 * t * t - 2.0 * t * t * t - u;
        let df = 6.0 * t - 6.0 * t * t;
        let step = if df.abs() > 1e-6 { f / df } else { 0.0 };
        t = (t - step).min(1.0 - 1e-7).max(1.0e-7);
    }
    t
}

/// Draw Y ∈ B(center,R) with pdf p(y) ∝ G_B^{3D}(x,y) (Green-ball importance sampling).
#[inline]
pub(crate) fn sample_ball_by_green_pdf(rng: &mut Rng, center: Vec3, radius: f32) -> Vec3 {
    let dir = sample_unit_sphere(rng); // uniform direction
    let u = rng.uniform_f32().max(1e-7); // radius via inverse CDF
    let t = inv_cdf_radius_under_green_pdf(u);
    center + dir * (radius * t)
}

/// Volume of a 3D ball of radius R.
#[inline]
pub(crate) fn ball_volume(radius: f32) -> f32 {
    (4.0 / 3.0) * PI * radius * radius * radius
}

/// Dirichlet Green's function on a ball in 3D
///
/// Green_B(x,y) for a ball B(x,R) centered at `x` with radius `R`, evaluated at distance `r=|y-x|`.
/// See Sawhney & Crane (2020), Appendix A (also reproduced in the WoSt paper):
/// G_B^3D(x,y) = (1 / 4π) * (1/r - 1/R), for 0 < r <= R.
///
/// Notes:
/// - Singular at r=0, but this happens with probability zero for continuous sampling; we clamp.
#[inline]
pub(crate) fn green_ball_3d(radius: f32, r: f32) -> f32 {
    let r = r.max(1e-7);
    (1.0 / (4.0 * PI)) * (1.0 / r - 1.0 / radius)
}

/// Total mass of the (Dirichlet) Green's function over a ball:  ∫_B G_B^3D(x,y) dy = R²/6.
#[inline]
pub(crate) fn green_ball_total_mass(radius: f32) -> f32 {
    (radius * radius) / 6.0
}

/// Walk-on-Spheres step: jump to a uniformly random point on the sphere `S(x, R)`.
#[inline]
pub(crate) fn wos_jump(x: Vec3, radius: f32, rng: &mut Rng) -> Vec3 {
    x + sample_unit_sphere(rng) * radius
}
