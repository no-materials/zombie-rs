//! Domain representations and closest-point queries.

use alloc::vec::Vec;
use libm::sqrtf;

use crate::math::{Vec3, closest_point_on_triangle};

/// Minimal information needed by WoS to define largest empty ball.
#[derive(Copy, Clone, Debug)]
pub struct Closest {
    /// Nearest point on the boundary.
    pub point: Vec3,
    /// Outward unit normal (best effort; for polygon soup we use facet normal at hit).
    pub normal: Vec3,
    /// Non-negative distance to boundary.
    pub distance: f32,
}

impl Closest {
    #[inline]
    pub fn empty() -> Self {
        Self {
            point: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            distance: f32::INFINITY,
        }
    }
}

/// A **volumetric domain Ω** supports queries from points in ℝ³ to their
/// distance and nearest boundary point on `∂Ω`. This is sufficient to run WoS.
///
/// Implementations:
/// - `PolygonSoupDomain` + `Bvh`
/// - `SdfDomain` (signed distance fields)
pub trait Domain: Send + Sync {
    /// Query the closest boundary point to `x`. Must satisfy:
    /// - `out.distance ≥ 0`
    /// - `out.point` is on `∂Ω`
    /// - `out.normal` is (approximately) outward unit normal
    fn closest(&self, x: Vec3) -> Closest;

    /// Is a point considered **inside** the domain Ω?
    ///
    /// Required for starting points. For SDFs this is `φ(x) < 0`; for polygon soup
    /// we can parity-count or rely on application context (here we use a ray parity
    /// heuristic; see implementation notes).
    fn is_inside(&self, x: Vec3) -> bool;
}

/// Wrap any signed distance function `φ: ℝ³→ℝ`. Inside is `φ(x) < 0`.
/// The normal is computed from the SDF gradient via central differences
/// (finite difference; stable if `φ` is well-behaved).
pub struct SdfDomain<F>
where
    F: Fn(Vec3) -> f32 + Send + Sync,
{
    phi: F,
    /// Step for numeric gradients.
    h: f32,
}

impl<F> SdfDomain<F>
where
    F: Fn(Vec3) -> f32 + Send + Sync,
{
    pub fn new(phi: F) -> Self {
        Self { phi, h: 1e-3 }
    }

    /// Set finite difference step for normals.
    pub fn with_step(self, h: f32) -> Self {
        Self { h, ..self }
    }

    #[inline]
    fn phi(&self, x: Vec3) -> f32 {
        (self.phi)(x)
    }

    #[inline]
    fn grad(&self, x: Vec3) -> Vec3 {
        let h = self.h;
        // Central finite differences (no heap, no branches)
        let dx = Vec3::new(h, 0.0, 0.0);
        let dy = Vec3::new(0.0, h, 0.0);
        let dz = Vec3::new(0.0, 0.0, h);
        let dphix = self.phi(x + dx) - self.phi(x - dx);
        let dphiy = self.phi(x + dy) - self.phi(x - dy);
        let dphiz = self.phi(x + dz) - self.phi(x - dz);
        Vec3::new(dphix, dphiy, dphiz) / (2.0 * h)
    }
}

impl<F> Domain for SdfDomain<F>
where
    F: Fn(Vec3) -> f32 + Send + Sync,
{
    #[inline]
    fn closest(&self, x: Vec3) -> Closest {
        // For an SDF, |φ(x)| is a conservative distance to the zero level.
        let d = self.phi(x);
        let n = self.grad(x).normalized();
        let p = x - n * d; // project onto level set along normal
        Closest {
            point: p,
            normal: n,
            distance: d.abs(),
        }
    }

    #[inline]
    fn is_inside(&self, x: Vec3) -> bool {
        self.phi(x) < 0.0
    }
}

/// Boolean combinators for SDFs (CSG-style)
///
/// These are standard Lipschitz-respecting compositions.
///
/// - **Union**: `min(φ₁, φ₂)`
/// - **Intersection**: `max(φ₁, φ₂)`
/// - **Difference**: `max(φ₁, −φ₂)`
///
/// Note: smooth blends can be added via soft-min/soft-max.
pub mod sdf_csg {
    use super::Vec3;

    /// Union φ = min(φ1, φ2)
    pub fn union<F1, F2>(phi1: F1, phi2: F2) -> impl Fn(Vec3) -> f32 + Send + Sync
    where
        F1: Fn(Vec3) -> f32 + Send + Sync,
        F2: Fn(Vec3) -> f32 + Send + Sync,
    {
        move |x| (phi1)(x).min((phi2)(x))
    }

    /// Intersection φ = max(φ1, φ2)
    pub fn intersection<F1, F2>(phi1: F1, phi2: F2) -> impl Fn(Vec3) -> f32 + Send + Sync
    where
        F1: Fn(Vec3) -> f32 + Send + Sync,
        F2: Fn(Vec3) -> f32 + Send + Sync,
    {
        move |x| (phi1)(x).max((phi2)(x))
    }

    /// Difference φ = max(φ1, -φ2)
    pub fn difference<F1, F2>(phi1: F1, phi2: F2) -> impl Fn(Vec3) -> f32 + Send + Sync
    where
        F1: Fn(Vec3) -> f32 + Send + Sync,
        F2: Fn(Vec3) -> f32 + Send + Sync,
    {
        move |x| (phi1)(x).max(-(phi2)(x))
    }
}

/// Polygon-soup (triangle mesh) domain.
///
/// Stores immutable vertex/triangle buffers and exposes a `closest()`
/// query accelerated by a BVH. “Inside” is determined by a robust, simple
/// parity test along a fixed ray (sufficient for many CAD uses; for general
/// non-manifold inputs, users may override with an application-specific predicate).
pub struct PolygonSoupDomain {
    pub(crate) verts: Vec<Vec3>,
    pub(crate) tris: Vec<[u32; 3]>,
    // Optional: precomputed facet normals to avoid recomputation
    pub(crate) tri_normals: Vec<Vec3>,
}

impl PolygonSoupDomain {
    /// Create a domain from positions and triangle indices.
    ///
    /// ### Invariants
    /// - Triangles index into `verts`.
    /// - Degenerate triangles are allowed but may affect normals; they are handled in closest-point.
    pub fn new(verts: Vec<Vec3>, tris: Vec<[u32; 3]>) -> Result<Self, &'static str> {
        // Validate indices
        let n = verts.len() as u32;
        for t in &tris {
            if t[0] >= n || t[1] >= n || t[2] >= n {
                return Err("triangle index out of bounds");
            }
        }
        // Precompute normals (best-effort; normalized or zero if degenerate)
        let mut tri_normals = Vec::with_capacity(tris.len());
        for t in &tris {
            let a = verts[t[0] as usize];
            let b = verts[t[1] as usize];
            let c = verts[t[2] as usize];
            let nrm = (b - a).cross(c - a).normalized();
            tri_normals.push(nrm);
        }
        Ok(Self {
            verts,
            tris,
            tri_normals,
        })
    }

    #[inline]
    pub(crate) fn tri(&self, i: usize) -> (Vec3, Vec3, Vec3) {
        let t = self.tris[i];
        (
            self.verts[t[0] as usize],
            self.verts[t[1] as usize],
            self.verts[t[2] as usize],
        )
    }
}

impl Domain for PolygonSoupDomain {
    #[inline]
    fn closest(&self, x: Vec3) -> Closest {
        // NOTE: For performance, users should prebuild a BVH and call `Bvh::closest_point`.
        // Here we expose a fallback O(n) scan for completeness. For real use, prefer the BVH path.
        let mut best = Closest::empty();
        let mut best_d2 = f32::INFINITY;
        for (i, t) in self.tris.iter().enumerate() {
            let a = self.verts[t[0] as usize];
            let b = self.verts[t[1] as usize];
            let c = self.verts[t[2] as usize];
            let q = closest_point_on_triangle(x, a, b, c);
            let d2 = (x - q).length_sq();
            if d2 < best_d2 {
                best_d2 = d2;
                best.point = q;
                best.distance = sqrtf(d2);
                best.normal = self.tri_normals[i];
            }
        }
        best
    }

    fn is_inside(&self, x: Vec3) -> bool {
        // Parity ray test along +X direction: count intersections with triangle planes.
        // This is a robust heuristic for many closed soups but not foolproof for all degeneracies.
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let mut hits = 0u32;

        for t in &self.tris {
            let a = self.verts[t[0] as usize];
            let b = self.verts[t[1] as usize];
            let c = self.verts[t[2] as usize];

            // Ray-triangle intersection test (Möller–Trumbore style)
            let eps = 1e-8f32;
            let e1 = b - a;
            let e2 = c - a;
            let pvec = dir.cross(e2);
            let det = e1.dot(pvec);
            if det.abs() < eps {
                continue;
            }
            let inv_det = 1.0 / det;
            let tvec = x - a;
            let u = tvec.dot(pvec) * inv_det;
            if u < 0.0 || u > 1.0 {
                continue;
            }
            let qvec = tvec.cross(e1);
            let v = dir.dot(qvec) * inv_det;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let tpar = e2.dot(qvec) * inv_det;
            if tpar >= 0.0 {
                hits ^= 1; // flip parity
            }
        }
        hits & 1 == 1
    }
}
