//! Random sampling utilities for spheres and balls.

use core::f32::consts::PI;
use libm::{cosf, coshf, powf, sinf, sinhf, sqrtf};

use crate::math::Vec3;
use crate::rng::Rng;

const INV_4PI: f32 = 0.079_577_47;

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

/// Draw Y ∈ B(center,R) with pdf `p(y) ∝ G_c(x,y)` for the screened Yukawa kernel (App. B.2).
///
/// The radial CDF is inverted via bisection to avoid numerical issues for small radii.
#[inline]
pub(crate) fn sample_ball_by_yukawa_pdf(rng: &mut Rng, center: Vec3, radius: f32, c: f32) -> Vec3 {
    if c.abs() < 1e-6 {
        return sample_ball_by_green_pdf(rng, center, radius);
    }
    let dir = sample_unit_sphere(rng);
    let u = rng.uniform_f32();
    let r = inv_cdf_radius_under_yukawa_pdf(u, radius, c);
    center + dir * r
}

/// Volume of a 3D ball of radius R.
#[inline]
pub(crate) fn ball_volume(radius: f32) -> f32 {
    (4.0 / 3.0) * PI * radius * radius * radius
}

/// Dirichlet Green's function on a ball in 3D
///
/// Green_B(x,y) for a ball B(x,R) centered at `x` with radius `R`, evaluated at distance `r=|y-x|`.
/// See Sawhney & Crane (2020), Appendix A (also reproduced in the WoSt paper: Appendix A: Eq. 26):
/// G_B^3D(x,y) = (1 / 4π) * (1/r - 1/R), for 0 < r <= R.
///
/// Notes:
/// - Singular at r=0, but this happens with probability zero for continuous sampling; we clamp.
#[inline]
pub(crate) fn green_ball_3d(radius: f32, r: f32) -> f32 {
    let rinv = 1.0 / r.max(1e-7);
    INV_4PI * (rinv - 1.0 / radius)
}

/// Total mass of the (Dirichlet) Green's function over a ball in 3d:  ∫_B G_B^3D(x,y) dy = R²/6.
///
/// Wost paper: Appendix A, Eq. 28.
#[inline]
pub(crate) fn green_ball_3d_total_mass(radius: f32) -> f32 {
    (radius * radius) / 6.0
}

/// Walk-on-Spheres step: jump to a uniformly random point on the sphere `S(x, R)`.
#[inline]
pub(crate) fn wos_jump(x: Vec3, radius: f32, rng: &mut Rng) -> Vec3 {
    x + sample_unit_sphere(rng) * radius
}

/// Yukawa Green's function in 3D (Appendix B.2): `G_c(x,y)` for radius `r = |x-y|` and ball radius `R`.
#[inline]
pub(crate) fn yukawa_green_3d(c: f32, r: f32, radius: f32) -> f32 {
    if c.abs() < 1e-6 {
        return green_ball_3d(radius, r);
    }
    let sqrt_c = sqrtf(c.max(0.0));
    let r = r.max(1e-6);
    let denom = r * sinhf(radius * sqrt_c);
    if denom.abs() < 1e-6 {
        green_ball_3d(radius, r)
    } else {
        let numer = sinhf((radius - r) * sqrt_c);
        INV_4PI * numer / denom
    }
}

/// Integral of Yukawa kernel over the ball (Appendix B.2).
#[inline]
pub(crate) fn yukawa_total_mass_3d(c: f32, radius: f32) -> f32 {
    if c.abs() < 1e-6 {
        return green_ball_3d_total_mass(radius);
    }
    let sqrt_c = sqrtf(c.max(0.0));
    let denom = sinhf(radius * sqrt_c);
    if denom.abs() < 1e-6 {
        green_ball_3d_total_mass(radius)
    } else {
        (1.0 / c) * (1.0 - radius * sqrt_c / denom)
    }
}

/// Normalization factor `C_3D = R sqrt(c) / sinh(R sqrt(c))` (Appendix B.2.1).
#[inline]
pub(crate) fn yukawa_normalization_3d(c: f32, radius: f32) -> f32 {
    if c.abs() < 1e-6 {
        return 1.0;
    }
    let sqrt_c = sqrtf(c.max(0.0));
    let denom = sinhf(radius * sqrt_c);
    if denom.abs() < 1e-6 {
        1.0
    } else {
        radius * sqrt_c / denom
    }
}

/// Invert the Yukawa radial CDF via bisection (Appendix B.2, Eq. 9).
#[inline]
fn inv_cdf_radius_under_yukawa_pdf(u: f32, radius: f32, c: f32) -> f32 {
    let u = u.min(1.0 - 1e-7).max(1e-7);
    let sqrt_c = sqrtf(c.max(0.0));
    if sqrt_c < 1e-6 {
        // Degenerates toward Laplace; reuse Green-ball sampler.
        let t = inv_cdf_radius_under_green_pdf(u);
        return radius * t;
    }

    let k = sqrt_c;
    let r_total = radius;
    let sinh_k_r = sinhf(k * r_total);
    let integral_total = (sinh_k_r / (k * k)) - (r_total / k);
    let target = u * integral_total;

    let mut lo = 0.0_f32;
    let mut hi = r_total;
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        let value = yukawa_partial_integral(mid, r_total, k, sinh_k_r);
        if value < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Evaluate ∫₀ʳ ρ sinh((R-ρ)√c) dρ used by the Yukawa CDF (Appendix B.2).
#[inline]
fn yukawa_partial_integral(r: f32, radius: f32, k: f32, sinh_k_r: f32) -> f32 {
    let diff = radius - r;
    let term2 = r * coshf(k * diff);
    let term3 = sinhf(k * diff);
    (sinh_k_r / (k * k)) - (term2 / k) - (term3 / (k * k))
}
