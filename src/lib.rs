#![no_std]
//! # `geom_kernel_wos.rs` — A `no_std`, safe, geometry kernel for grid-free Monte-Carlo geometry processing
//!
//! This file implements core abstractions and a reference implementation of
//! **Walk-on-Spheres (WoS)** estimators for **Laplace** and **(scaffold for) Poisson** problems
//! on **volumetric domains** without a background grid/mesh. The design is informed by the paper:
//!
//! *Sawhney & Crane (TOG, 2020): “Monte Carlo Geometry Processing: A Grid-Free Approach to PDE-Based Methods on Volumetric Domains”*.
//!
//! ## What this kernel provides
//!
//! - **Domain abstraction** via trait bounds to support multiple boundary representations:
//!   - Polygon soup (triangle meshes) with **closest-point** queries accelerated by a **BVH**.
//!   - Implicit surfaces via **signed distance fields (SDFs)**.
//!   - **Boolean combinations** of SDFs (CSG-style) by composing distance functions.
//! - **Safe, `no_std`**, allocation-enabled (via `alloc`) Rust with a focus on:
//!   - **Performance** (cache-friendly BVH, no virtual dispatch in hot loops, inlined math).
//!   - **Ergonomics** (clear types, enforced invariants).
//!   - **Future multi-threading** (all public types are `Send + Sync` when their generics are; no global mutable state).
//! - **WoS Laplace–Dirichlet estimator**: unbiased, grid-free evaluation `u(x)` at arbitrary points,
//!   with one parameter **stopping tolerance** `ε`.
//! - **Hooks for Poisson** estimators with user-provided source models (kept minimal here, to be extended).
//!
//! ## Why Walk-on-Spheres (WoS)?
//! WoS replaces global linear solves with **independent random walks** that jump from a point to the
//! boundary of the **largest empty ball** inside the domain until they are within `ε` of the boundary,
//! where the **Dirichlet boundary condition** is sampled. This yields an **unbiased** estimator of
//! harmonic (Laplace) solutions, with variance ∝ `1/N` for `N` walks, and **no discretization error**
//! from gridding/meshing the volume. See §2 and §5 in the paper.
//!
//! ## Key technical terms
//! - **Dirichlet boundary condition**: the solution value is prescribed on the boundary `∂Ω`.
//! - **Largest empty ball radius at x**: distance from point `x` to the boundary; radius `R(x)`.
//! - **Closest-point query**: find nearest point on `∂Ω` to a given `x` and its surface normal.
//! - **BVH (Bounding Volume Hierarchy)**: tree of AABBs to accelerate closest-point/triangle tests.
//! - **SDF (Signed Distance Field)**: function `φ: ℝ³→ℝ` with value = signed distance to surface; `|φ|` is distance.
//!
//! ## Safety & invariants
//! - The kernel is fully **safe Rust**. No `unsafe` blocks.
//! - Geometric invariants are enforced by type wrappers and constructor checks (e.g., normalized normals).
//! - BVH node arrays are immutable after build, enabling `Send + Sync` usage out of the box.
//!
//! ## `no_std` and allocation
//! This crate uses `core` and `alloc` only. To use collections (`Vec`, `Box`) in `no_std`, your
//! environment must provide a global allocator.
//!
//! ## Extending this kernel
//! - Add Poisson estimators.
//! - Add GPU/vectorized closest-point backends.
//! - Add Neumann/Robin boundary conditions and mixed problems.
//! - Add gradients if not already available.
//! - Add more domain types (e.g., NURBS, point clouds).
//! - Add bias-variance reduction (control variates), progressive / adaptive sampling, denoising hooks.
//!
//! ---
//!
//! ### Quick start (sketch)
//! ```ignore
//! // Build a polygon-soup domain:
//! let domain = PolygonSoupDomain::new(vertices, triangles).expect("valid mesh");
//! let bvh = Bvh::build(&domain); // O(n log n)
//!
//! // Define a Dirichlet boundary condition function g(p): ℝ³→ℝ
//! let bc = BoundaryDirichletFn::new(|p: Vec3| p.x + p.y + p.z);
//!
//! // Configure WoS
//! let params = WosParams::new(1e-4, 10_000); // ε, max_steps
//! let mut rng = XorShift64::seed_from(123456789);
//!
////! // Evaluate u(x) for Laplace with Dirichlet data
//! let u = wos_laplace_dirichlet(&domain, &bvh, &bc, params, &mut rng, Vec3::new(0.1, 0.2, 0.3));
//! ```
//!
//! ---

extern crate alloc;

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::f32::consts::PI;
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use libm::{cosf, sinf, sqrtf};

/// ### `math` — Minimal 3D math (no_std)
///
/// Lightweight, inlined, panic-free vector math tailored for geometry kernels.
/// We keep it deliberately small and explicit for clarity and performance.
mod math {
    use libm::sqrtf;

    use super::*;

    /// 3D vector with `f32` components.
    ///
    /// - Invariants: none beyond `f32` domain; normalization is explicit.
    /// - Why `f32`? It’s commonly faster and sufficient for WoS sampling; consider `f64` feature if needed.
    #[derive(Copy, Clone, Default, PartialEq)]
    pub struct Vec3 {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    impl fmt::Debug for Vec3 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Vec3({:.6}, {:.6}, {:.6})", self.x, self.y, self.z)
        }
    }

    impl Vec3 {
        #[inline(always)]
        pub const fn new(x: f32, y: f32, z: f32) -> Self {
            Self { x, y, z }
        }

        #[inline(always)]
        pub fn dot(self, rhs: Self) -> f32 {
            self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
        }

        #[inline(always)]
        pub fn length(self) -> f32 {
            sqrtf(self.dot(self))
        }

        #[inline(always)]
        pub fn length_sq(self) -> f32 {
            self.dot(self)
        }

        #[inline(always)]
        pub fn normalized(self) -> Self {
            let n = self.length();
            if n > 0.0 { self / n } else { self }
        }

        #[inline(always)]
        pub fn cross(self, rhs: Self) -> Self {
            Self {
                x: self.y * rhs.z - self.z * rhs.y,
                y: self.z * rhs.x - self.x * rhs.z,
                z: self.x * rhs.y - self.y * rhs.x,
            }
        }

        #[inline(always)]
        pub fn min(self, rhs: Self) -> Self {
            Self {
                x: self.x.min(rhs.x),
                y: self.y.min(rhs.y),
                z: self.z.min(rhs.z),
            }
        }

        #[inline(always)]
        pub fn max(self, rhs: Self) -> Self {
            Self {
                x: self.x.max(rhs.x),
                y: self.y.max(rhs.y),
                z: self.z.max(rhs.z),
            }
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            Self::new(self.x.abs(), self.y.abs(), self.z.abs())
        }
    }

    impl Add for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self::Output {
            Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
        }
    }
    impl AddAssign for Vec3 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }
    impl Sub for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self::Output {
            Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
    impl SubAssign for Vec3 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }
    impl Mul<f32> for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: f32) -> Self::Output {
            Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
        }
    }
    impl Div<f32> for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: f32) -> Self::Output {
            Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
        }
    }
    impl Neg for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self::Output {
            Self::new(-self.x, -self.y, -self.z)
        }
    }

    /// Axis-aligned bounding box.
    ///
    /// Used in the BVH to bound triangle primitives for fast rejection in closest-point queries.
    #[derive(Copy, Clone, Debug)]
    pub struct Aabb {
        pub min: Vec3,
        pub max: Vec3,
    }

    impl Aabb {
        #[inline]
        pub fn empty() -> Self {
            let inf = f32::INFINITY;
            let ninf = f32::NEG_INFINITY;
            Self {
                min: Vec3::new(inf, inf, inf),
                max: Vec3::new(ninf, ninf, ninf),
            }
        }

        #[inline]
        pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Self {
            let min = a.min(b).min(c);
            let max = a.max(b).max(c);
            Self { min, max }
        }

        #[inline]
        pub fn grow(&mut self, p: Vec3) {
            self.min = self.min.min(p);
            self.max = self.max.max(p);
        }

        #[inline]
        pub fn union(self, other: Self) -> Self {
            Self {
                min: self.min.min(other.min),
                max: self.max.max(other.max),
            }
        }

        #[inline]
        pub fn longest_axis(&self) -> usize {
            let d = self.max - self.min;
            if d.x >= d.y && d.x >= d.z {
                0
            } else if d.y >= d.z {
                1
            } else {
                2
            }
        }

        /// Squared distance from a point to this AABB (useful for early-out pruning).
        #[inline]
        pub fn distance_sq(&self, p: Vec3) -> f32 {
            let clamped = Vec3::new(
                p.x.max(self.min.x).min(self.max.x),
                p.y.max(self.min.y).min(self.max.y),
                p.z.max(self.min.z).min(self.max.z),
            );
            (p - clamped).length_sq()
        }
    }

    /// Compute closest point on a triangle (a,b,c) to point p.
    ///
    /// Reference: “Real-Time Collision Detection” (Christer Ericson).
    #[inline]
    pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
        // Check vertex regions
        let ab = b - a;
        let ac = c - a;
        let ap = p - a;
        let d1 = ab.dot(ap);
        let d2 = ac.dot(ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return a;
        }

        // Check vertex region B
        let bp = p - b;
        let d3 = ab.dot(bp);
        let d4 = ac.dot(bp);
        if d3 >= 0.0 && d4 <= d3 {
            return b;
        }

        // Check edge AB
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return a + ab * v;
        }

        // Check vertex region C
        let cp = p - c;
        let d5 = ab.dot(cp);
        let d6 = ac.dot(cp);
        if d6 >= 0.0 && d5 <= d6 {
            return c;
        }

        // Check edge AC
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return a + ac * w;
        }

        // Check edge BC
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + (c - b) * w;
        }

        // Inside face region: barycentric projection
        let n = ab.cross(ac);
        let nlen2 = n.length_sq();
        if nlen2 > 0.0 {
            let distance = (p - a).dot(n) / sqrtf(nlen2);
            return p - n.normalized() * distance;
        }
        // Degenerate triangle: return nearest vertex
        let d_a = (p - a).length_sq();
        let d_b = (p - b).length_sq();
        let d_c = (p - c).length_sq();
        if d_a <= d_b && d_a <= d_c {
            a
        } else if d_b <= d_c {
            b
        } else {
            c
        }
    }
}
use math::*;

/// ### Random number generation (`no_std`)
///
/// XorShift64* style RNG for reproducible, fast pseudorandom sampling.
/// This is sufficient for Monte Carlo exploration and WoS jumps.
mod rng {
    /// A small, fast 64-bit XorShift PRNG.
    ///
    /// - `no_std` friendly.
    /// - Not cryptographically secure.
    #[derive(Clone)]
    pub struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        /// Seed the RNG. A zero seed is remapped to a non-zero constant to avoid the fixed point.
        #[inline]
        pub fn seed_from(seed: u64) -> Self {
            let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
            Self { state: s }
        }

        #[inline]
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            // xorshift64* variant
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        /// Uniform in [0,1).
        #[inline]
        pub fn uniform_f32(&mut self) -> f32 {
            const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);
            let v = (self.next_u64() >> 32) as u32;
            (v as f32) * SCALE
        }
    }

    impl Default for XorShift64 {
        fn default() -> Self {
            Self::seed_from(0xA5A5_A5A5_1234_5678)
        }
    }

    pub use XorShift64 as Rng;
}
use rng::Rng;

/// ### Boundary condition abstraction
///
/// Encodes Dirichlet data `g(p)` sampled on `∂Ω`.
pub trait BoundaryDirichlet: Send + Sync {
    /// Evaluate boundary value at a boundary point `p ∈ ∂Ω`.
    ///
    /// This is called only when the walk terminates within `ε` of the boundary;
    /// `p` should be understood as the nearest boundary point.
    fn value(&self, p: Vec3) -> f32;
}

/// Simple functional wrapper implementing `BoundaryDirichlet`.
pub struct BoundaryDirichletFn<F>
where
    F: Fn(Vec3) -> f32 + Send + Sync,
{
    f: F,
}
impl<F> BoundaryDirichletFn<F>
where
    F: Fn(Vec3) -> f32 + Send + Sync,
{
    pub fn new(f: F) -> Self {
        Self { f }
    }
}
impl<F> BoundaryDirichlet for BoundaryDirichletFn<F>
where
    F: Fn(Vec3) -> f32 + Send + Sync,
{
    #[inline]
    fn value(&self, p: Vec3) -> f32 {
        (self.f)(p)
    }
}

/// ### Closest-point query result
///
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

/// ### Domain interface
///
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

/// ### SDF Domain
///
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

/// ### Boolean combinators for SDFs (CSG-style)
///
/// These are standard Lipschitz-respecting compositions.
///
/// - **Union**: `min(φ₁, φ₂)`
/// - **Intersection**: `max(φ₁, φ₂)`
/// - **Difference**: `max(φ₁, −φ₂)`
///
/// Note: smooth blends can be added via soft-min/soft-max.
pub mod sdf_csg {
    use super::*;

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

/// ### Polygon-soup (triangle mesh) domain
///
/// Stores immutable vertex/triangle buffers and exposes a `closest()`
/// query accelerated by a BVH. “Inside” is determined by a robust, simple
/// parity test along a fixed ray (sufficient for many CAD uses; for general
/// non-manifold inputs, users may override with an application-specific predicate).
pub struct PolygonSoupDomain {
    verts: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    // Optional: precomputed facet normals to avoid recomputation
    tri_normals: Vec<Vec3>,
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
    fn tri(&self, i: usize) -> (Vec3, Vec3, Vec3) {
        let t = self.tris[i];
        (
            self.verts[t[0] as usize],
            self.verts[t[1] as usize],
            self.verts[t[2] as usize],
        )
    }
}

/// A single BVH node.
/// We store a binary tree with nodes laid out in a flat array for cache locality.
#[derive(Copy, Clone, Debug)]
struct BvhNode {
    aabb: Aabb,
    // If leaf: range over primitive indices [start, count); if inner: left/right child indices
    start: u32,
    count: u32,
    left: u32,
    right: u32,
    is_leaf: bool,
}

/// BVH over triangle indices for closest-point acceleration.
///
/// Build: longest-axis median split. Query: branch-and-bound over leaf triangles.
pub struct Bvh {
    nodes: Vec<BvhNode>,
    prims: Vec<u32>, // permutation of triangle indices
}

impl Bvh {
    /// Build a BVH for a polygon soup domain.
    pub fn build(mesh: &PolygonSoupDomain) -> Self {
        // Primitive bounds: triangle AABBs
        let mut prims: Vec<u32> = (0..mesh.tris.len() as u32).collect();
        let mut nodes: Vec<BvhNode> = Vec::new();
        nodes.reserve(mesh.tris.len() * 2);

        fn tri_aabb(mesh: &PolygonSoupDomain, i: usize) -> Aabb {
            let (a, b, c) = mesh.tri(i);
            Aabb::from_points(a, b, c)
        }

        fn build_rec(mesh: &PolygonSoupDomain, nodes: &mut Vec<BvhNode>, prims: &mut [u32]) -> u32 {
            // Compute node bounds
            let mut node_aabb = Aabb::empty();
            for &pi in prims.iter() {
                node_aabb = node_aabb.union(tri_aabb(mesh, pi as usize));
            }

            let node_index = nodes.len() as u32;
            // Leaf threshold: small ranges
            if prims.len() <= 8 {
                nodes.push(BvhNode {
                    aabb: node_aabb,
                    start: 0, // filled later
                    count: prims.len() as u32,
                    left: 0,
                    right: 0,
                    is_leaf: true,
                });
                return node_index;
            }

            // Split along longest axis by median on primitive centroids
            let axis = node_aabb.longest_axis();
            prims.sort_unstable_by(|&a, &b| {
                let ca = centroid(mesh, a as usize);
                let cb = centroid(mesh, b as usize);
                let oa = match axis {
                    0 => ca.x,
                    1 => ca.y,
                    _ => ca.z,
                };
                let ob = match axis {
                    0 => cb.x,
                    1 => cb.y,
                    _ => cb.z,
                };
                oa.partial_cmp(&ob).unwrap_or(Ordering::Equal)
            });
            let mid = prims.len() / 2;
            let (left_slice, right_slice) = prims.split_at_mut(mid);

            // Placeholder, to be patched after recursion:
            let dummy = BvhNode {
                aabb: node_aabb,
                start: 0,
                count: 0,
                left: 0,
                right: 0,
                is_leaf: false,
            };
            nodes.push(dummy);

            let left_index = build_rec(mesh, nodes, left_slice);
            let right_index = build_rec(mesh, nodes, right_slice);

            // Patch internal node
            let node = &mut nodes[node_index as usize];
            node.left = left_index;
            node.right = right_index;

            node_index
        }

        fn centroid(mesh: &PolygonSoupDomain, i: usize) -> Vec3 {
            let (a, b, c) = mesh.tri(i);
            (a + b + c) / 3.0
        }

        // Build hierarchy
        let root = build_rec(mesh, &mut nodes, &mut prims[..]);

        // Flatten leaf ranges: assign contiguous ranges in `prims`
        // We perform a DFS to remap leaf prims into contiguous order and set start/count.
        let mut remap: Vec<u32> = Vec::with_capacity(prims.len());
        fn assign_ranges(
            mesh: &PolygonSoupDomain,
            nodes: &mut [BvhNode],
            prims: &mut [u32],
            remap: &mut Vec<u32>,
            idx: u32,
        ) {
            let node = nodes[idx as usize];
            if node.is_leaf {
                let start = remap.len() as u32;
                remap.extend_from_slice(prims);
                let count = (remap.len() as u32) - start;
                let mut n = node;
                n.start = start;
                n.count = count;
                nodes[idx as usize] = n;
            } else {
                // Split prims same way as build to maintain correspondence
                let axis = node.aabb.longest_axis();
                prims.sort_unstable_by(|&a, &b| {
                    let ca = {
                        let (pa, pb, pc) = mesh.tri(a as usize);
                        (pa + pb + pc) / 3.0
                    };
                    let cb = {
                        let (pa, pb, pc) = mesh.tri(b as usize);
                        (pa + pb + pc) / 3.0
                    };
                    let oa = match axis {
                        0 => ca.x,
                        1 => ca.y,
                        _ => ca.z,
                    };
                    let ob = match axis {
                        0 => cb.x,
                        1 => cb.y,
                        _ => cb.z,
                    };
                    oa.partial_cmp(&ob).unwrap_or(Ordering::Equal)
                });
                let mid = prims.len() / 2;
                let (left_slice, right_slice) = prims.split_at_mut(mid);
                assign_ranges(mesh, nodes, left_slice, remap, nodes[idx as usize].left);
                assign_ranges(mesh, nodes, right_slice, remap, nodes[idx as usize].right);
            }
        }
        assign_ranges(mesh, &mut nodes[..], &mut prims[..], &mut remap, root);

        Self {
            nodes,
            prims: remap,
        }
    }

    /// Closest point on the boundary for polygon soup.
    ///
    /// Returns `(Closest, best_tri_index)`.
    pub fn closest_point(&self, mesh: &PolygonSoupDomain, x: Vec3) -> (Closest, u32) {
        // Stack-based traversal (no recursion)
        let mut stack: [u32; 64] = [0; 64];
        let mut sp = 0usize;
        if self.nodes.is_empty() {
            return (Closest::empty(), u32::MAX);
        }
        stack[sp] = 0;
        sp += 1;

        let mut best_d2 = f32::INFINITY;
        let mut best_point = Vec3::new(0.0, 0.0, 0.0);
        let mut best_nrm = Vec3::new(0.0, 1.0, 0.0);
        let mut best_tri = u32::MAX;

        while sp > 0 {
            sp -= 1;
            let idx = stack[sp];
            let node = self.nodes[idx as usize];
            // Prune by AABB distance
            if node.aabb.distance_sq(x) >= best_d2 {
                continue;
            }
            if node.is_leaf {
                let start = node.start as usize;
                let end = start + node.count as usize;
                for i in start..end {
                    let tri_idx = self.prims[i] as usize;
                    let (a, b, c) = mesh.tri(tri_idx);
                    let q = math::closest_point_on_triangle(x, a, b, c);
                    let d2 = (x - q).length_sq();
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_point = q;
                        best_nrm = mesh.tri_normals[tri_idx]; // facet normal (approx.)
                        best_tri = tri_idx as u32;
                    }
                }
            } else {
                // Visit nearer child first (AABB heuristic)
                let l = node.left;
                let r = node.right;
                let dl = self.nodes[l as usize].aabb.distance_sq(x);
                let dr = self.nodes[r as usize].aabb.distance_sq(x);
                if dl < dr {
                    stack[sp] = r;
                    sp += 1;
                    stack[sp] = l;
                    sp += 1;
                } else {
                    stack[sp] = l;
                    sp += 1;
                    stack[sp] = r;
                    sp += 1;
                }
            }
        }

        let dist = sqrtf(best_d2);
        (
            Closest {
                point: best_point,
                normal: best_nrm,
                distance: dist,
            },
            best_tri,
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
            let q = math::closest_point_on_triangle(x, a, b, c);
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

/// ### WoS configuration
///
/// - `epsilon` (`ε`): stopping tolerance; when within `ε` of the boundary, we terminate and sample `g`.
/// - `max_steps`: hard cap to avoid infinite loops in degenerate setups (very rare with proper geometry).
#[derive(Copy, Clone, Debug)]
pub struct WosParams {
    pub epsilon: f32,
    pub max_steps: u32,
}
impl WosParams {
    pub fn new(epsilon: f32, max_steps: u32) -> Self {
        Self { epsilon, max_steps }
    }
}

/// ### Uniform sampling on the unit sphere (S²)
///
/// Uses a branch-free method based on two uniforms.
#[inline]
fn sample_unit_sphere(rng: &mut Rng) -> Vec3 {
    // Marsaglia method
    let u = 2.0 * rng.uniform_f32() - 1.0; // z in [-1,1]
    let t = 2.0 * PI * rng.uniform_f32();
    let r = sqrtf(1.0 - u * u);
    Vec3::new(r * cosf(t), r * sinf(t), u)
}

/// ### WoS step
///
/// Given a point `x` and a radius `R` to the nearest boundary, jump to a uniformly random point
/// on the sphere `S(x, R)`.
#[inline]
fn wos_jump(x: Vec3, radius: f32, rng: &mut Rng) -> Vec3 {
    x + sample_unit_sphere(rng) * radius
}

/// ### Laplace (Dirichlet) estimator via WoS
///
/// Unbiased estimator of the harmonic function `u` with boundary data `g` on `∂Ω`.
///
/// **Algorithm:** Starting at `x ∈ Ω`, repeat:
/// 1. Query `R = dist(x, ∂Ω)`.
/// 2. If `R <= ε`: return `g(p_closest)`.
/// 3. Else: set `x ← x + R * ξ` where `ξ ~ Uniform(S²)` and loop.
///
/// See: paper §2.2 (Walk on Spheres), §6.1 (stopping tolerance).
pub fn wos_laplace_dirichlet<D: Domain, G: BoundaryDirichlet>(
    domain: &D,
    accel: &impl ClosestAccel<D>,
    g: &G,
    params: WosParams,
    rng: &mut Rng,
    mut x: Vec3,
) -> f32 {
    let mut steps = 0u32;
    // Optional safety: reject if starting outside
    debug_assert!(
        domain.is_inside(x),
        "wos_laplace_dirichlet: x must start inside Ω"
    );

    loop {
        let c = accel.closest(domain, x);
        if c.distance <= params.epsilon {
            // Terminate: sample Dirichlet boundary value at nearest boundary point
            return g.value(c.point);
        }
        x = wos_jump(x, c.distance, rng);
        steps += 1;
        if steps >= params.max_steps {
            // Failsafe: sample boundary and return (introduces small bias only if hit)
            let c = accel.closest(domain, x);
            return g.value(c.point);
        }
    }
}

/// ### Closest-point accelerator trait
///
/// Allows plugging in different acceleration structures (e.g., BVH, spatial hashing).
pub trait ClosestAccel<D: Domain>: Send + Sync {
    fn closest(&self, domain: &D, x: Vec3) -> Closest;
}

/// A no-acceleration adapter (O(n) scan) for any `Domain`.
pub struct ClosestNaive;
impl<D: Domain> ClosestAccel<D> for ClosestNaive {
    #[inline]
    fn closest(&self, domain: &D, x: Vec3) -> Closest {
        domain.closest(x)
    }
}

/// BVH accelerator for `PolygonSoupDomain`.
pub struct BvhAccel<'a> {
    bvh: &'a Bvh,
    mesh: &'a PolygonSoupDomain,
}
impl<'a> BvhAccel<'a> {
    pub fn new(bvh: &'a Bvh, mesh: &'a PolygonSoupDomain) -> Self {
        Self { bvh, mesh }
    }
}
impl<'a> ClosestAccel<PolygonSoupDomain> for BvhAccel<'a> {
    #[inline]
    fn closest(&self, _domain: &PolygonSoupDomain, x: Vec3) -> Closest {
        self.bvh.closest_point(self.mesh, x).0
    }
}

/// ### (Scaffold) Poisson estimator
///
/// The classical WoS is exact for Laplace problems. For `Δu = f` (Poisson) in ℝ³,
/// unbiased estimators integrate the source via Green’s function or related
/// randomization tricks (see paper §2.3, §3 for derivatives/estimators).
///
/// Here we provide a **hook** that allows users to inject a problem-specific integral
/// estimator for the source term while reusing WoS skeleton for boundary handling.
///
/// The default implementation **returns the Laplace solution** (i.e., ignores `f`)
/// and serves as a placeholder. Extend this function to implement your preferred
/// source estimator (e.g., control variates, Russian roulette path sampling).
pub trait SourceTerm: Send + Sync {
    /// Source `f(x)` in `Δu = f`.
    fn value(&self, x: Vec3) -> f32;
}

pub fn wos_poisson_dirichlet<D: Domain, G: BoundaryDirichlet, F: SourceTerm>(
    domain: &D,
    accel: &impl ClosestAccel<D>,
    g: &G,
    _f: &F,
    params: WosParams,
    rng: &mut Rng,
    x: Vec3,
) -> f32 {
    // Placeholder: return Laplace solution. Replace by integrating f along the walk
    // with appropriate kernel (e.g., free-space Green’s function) and/or control variates.
    wos_laplace_dirichlet(domain, accel, g, params, rng, x)
}

mod tests {
    use crate::*;

    #[test]
    fn smoke() {
        // Define a sphere SDF of radius 1 at origin:
        let sphere = SdfDomain::new(|p: Vec3| p.length() - 1.0);

        // Dirichlet boundary: u = 1 on boundary
        let bc = BoundaryDirichletFn::new(|_p| 1.0);

        // WoS config
        let params = WosParams::new(1e-4, 10_000);
        let mut rng = rng::Rng::seed_from(42);

        // Evaluate near center (harmonic extension is constant 1.0 for this bc)
        let u = wos_laplace_dirichlet(
            &sphere,
            &ClosestNaive,
            &bc,
            params,
            &mut rng,
            Vec3::new(0.0, 0.0, 0.0),
        );
        assert!((u - 1.0).abs() < 5e-2); // stochastic tolerance
    }
}
