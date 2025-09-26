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
//! - Support more PDEs.
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
use libm::{cosf, powf, sinf, sqrtf};

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
fn sample_ball_by_green_pdf(rng: &mut Rng, center: Vec3, radius: f32) -> Vec3 {
    let dir = sample_unit_sphere(rng); // uniform direction
    let u = rng.uniform_f32().max(1e-7); // radius via inverse CDF
    let t = inv_cdf_radius_under_green_pdf(u);
    center + dir * (radius * t)
}

/// Total mass of the (Dirichlet) Green's function over a ball:  ∫_B G_B^3D(x,y) dy = R²/6.
#[inline]
fn green_ball_total_mass(radius: f32) -> f32 {
    (radius * radius) / 6.0
}

/// Draw Y ∈ B(center,R) uniformly (volume pdf).
#[inline]
fn sample_ball_uniform(rng: &mut Rng, center: Vec3, radius: f32) -> Vec3 {
    let dir = sample_unit_sphere(rng); // S²
    let u = rng.uniform_f32().max(1e-7); // radius ~ R * u^(1/3)
    center + dir * (radius * powf(u, 1.0 / 3.0))
}

/// Volume of a 3D ball of radius R.
#[inline]
fn ball_volume(radius: f32) -> f32 {
    (4.0 / 3.0) * PI * radius * radius * radius
}

/// ### Dirichlet Green's function on a ball in 3D
///
/// Green_B(x,y) for a ball B(x,R) centered at `x` with radius `R`, evaluated at distance `r=|y-x|`.
/// See Sawhney & Crane (2020), Appendix A (also reproduced in the WoSt paper):
/// G_B^3D(x,y) = (1 / 4π) * (1/r - 1/R), for 0 < r <= R.
///
/// Notes:
/// - Singular at r=0, but this happens with probability zero for continuous sampling; we clamp.
#[inline]
fn green_ball_3d(radius: f32, r: f32) -> f32 {
    let r = r.max(1e-7);
    (1.0 / (4.0 * PI)) * (1.0 / r - 1.0 / radius)
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

/// ### Poisson estimator settings
///
/// Unbiased estimator for the Dirichlet Poisson problem **in 3D** using WoS + per-step
/// Monte Carlo integration inside each largest empty ball.
///
/// **Convention solved here:** `-Δ u = f` in Ω, `u = g` on ∂Ω.  
/// (This is the common potential-theory sign so that a constant positive source yields
/// a positive solution inside a unit ball.)
///
/// The estimator adds, at *each* WoS step k with ball radius R_k:
///   I_k ≈ Vol(B_k) * (1/M) Σ_i G_B^3D(R_k, |Y_i - x_k|) * f(Y_i),
/// where `Y_i ~ Uniform(B_k)`. Summing I_k over steps is an unbiased estimate of the
/// domain integral term; termination on ∂Ω yields the boundary term `g`.
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

/// Source term in the PDE **-Δ u = f** (3D).
///
/// If your PDE is written as `Δ u = f`, pass `-f` here.
///
/// `point_params()` is an *optional* hint. By default it returns `None`.
/// Implementations representing a single Dirac source should override it
/// to return `Some((position, strength))`.
pub trait SourceTerm: Send + Sync {
    /// Evaluates a *regular* source at `x`. For a pure Dirac delta, this should return 0.
    fn value(&self, x: Vec3) -> f32;

    /// Optional: return `(z, q)` if `f(y) = q · δ(y − z)`.
    #[inline]
    fn point_params(&self) -> Option<(Vec3, f32)> {
        None
    }
}

/// A concrete δ (Dirac) source:  f(y) = q · δ(y − z)
pub struct PointSource {
    pub position: Vec3,
    pub strength: f32,
}
impl SourceTerm for PointSource {
    /// δ is not sampleable by point eval
    #[inline]
    fn value(&self, _x: Vec3) -> f32 {
        0.0
    }
    #[inline]
    fn point_params(&self) -> Option<(Vec3, f32)> {
        Some((self.position, self.strength))
    }
}

/// ### Poisson (Dirichlet) estimator via Walk-on-Spheres + ball Green function
///
/// Solves the **3D** Poisson problem with Dirichlet boundary data `g`:
/// ```text
///   -Δ u = f   in Ω,
///        u = g on ∂Ω.
/// ```
/// Returns a **single-path** unbiased Monte Carlo estimate of `u(x)` using:
/// - **WoS** for boundary hitting,
/// - **Uniform-in-ball sampling** inside each largest empty ball to estimate the
///   domain integral with the **Dirichlet Green function on a ball**.
///
/// Variance controls:
/// - Increase `poisson.interior_samples_per_step`.
/// - Average multiple independent calls (standard MC `1/N` variance).
///
/// Future extensions:
/// - Screened Poisson: multiply per-step weights by `Q_σ,B` (§A.2 in WoSt paper).
/// - Control variates (§4 in MC-Geometry paper).
#[allow(non_snake_case)]
pub fn wos_poisson_dirichlet<D, G, F>(
    domain: &D,
    accel: &impl ClosestAccel<D>,
    g: &G,
    fsrc: &F,
    params: WosParams,
    poisson: PoissonParams,
    rng: &mut Rng,
    mut x: Vec3,
) -> f32
where
    D: Domain,
    G: BoundaryDirichlet,
    F: SourceTerm,
{
    debug_assert!(
        domain.is_inside(x),
        "wos_poisson_dirichlet: x must be inside Ω"
    );

    let mut steps = 0u32;
    let mut acc = 0.0_f32;

    // Optional point source
    let point = fsrc.point_params();

    loop {
        let c = accel.closest(domain, x);
        let R = c.distance;

        if R <= params.epsilon {
            return acc + g.value(c.point);
        }

        // -------- Point-source analytic contribution (if present)
        if let Some((z, q)) = point {
            let r = (z - x).length();
            if r <= R {
                // The δ integrates exactly over the ball: ∫_B G δ = G_B(x,z)
                acc += q * green_ball_3d(R, r.max(poisson.min_r));
                // Note: keep walking; other balls may contribute nothing for this δ.
            }
        }

        // -------- Volume term for general sources
        let m = poisson.interior_samples_per_step.max(1);
        match poisson.sampling {
            InteriorSampling::Uniform => {
                // I ≈ Vol(B) * (1/m) Σ G_B(x,Y) f(Y),  Y ~ Uniform(B)
                let vol = ball_volume(R);
                let inv_m = 1.0 / (m as f32);
                let mut sum = 0.0;
                for _ in 0..m {
                    let y = sample_ball_uniform(rng, x, R);
                    let r = (y - x).length().max(poisson.min_r);
                    let gB = green_ball_3d(R, r);
                    // For point sources, f.value(y)=0 almost surely, which is fine;
                    // the analytic kick above already handled the δ contribution.
                    sum += gB * fsrc.value(y);
                }
                acc += vol * (sum * inv_m);
            }
            InteriorSampling::GreenBall => {
                // With p(y) ∝ G_B, the integral becomes Z * E[f(Y)] with Z = R^2/6.
                // This eliminates the G_B factor from the MC sum (nice variance cut).
                let Z = green_ball_total_mass(R);
                let inv_m = 1.0 / (m as f32);
                let mut sum = 0.0;
                for _ in 0..m {
                    let y = sample_ball_by_green_pdf(rng, x, R);
                    sum += fsrc.value(y);
                }
                acc += Z * (sum * inv_m);
            }
        }

        // Step
        x = wos_jump(x, R, rng);

        steps += 1;
        if steps >= params.max_steps {
            let c = accel.closest(domain, x);
            return acc + g.value(c.point);
        }
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

/// Generic single-ball surface-term gradient estimator:
/// returns (3/R) * E[ U(x+R ξ) * ξ ], where U is "how to evaluate u(·)".
fn grad_surface_single_ball<D, A, Ueval>(
    domain: &D,
    accel: &A,
    eval_u: &mut Ueval,
    walk: WosParams,
    boundary_dirs: u32,
    min_r: f32,
    rng: &mut Rng,
    x: Vec3,
) -> Vec3
where
    D: Domain,
    A: ClosestAccel<D>,
    Ueval: FnMut(&D, &A, WosParams, &mut Rng, Vec3) -> f32,
{
    let c = accel.closest(domain, x);
    let R = (c.distance).max(min_r);
    let m = boundary_dirs.max(1);
    let mut acc = Vec3::new(0.0, 0.0, 0.0);
    for _ in 0..m {
        let xi = sample_unit_sphere(rng);
        let xp = x + xi * R;
        let up = eval_u(domain, accel, walk, rng, xp);
        acc += xi * up;
    }
    acc * (3.0 / (R * m as f32))
}

/// ### Gradient of the Laplace–Dirichlet solution via a single-ball estimator.
///
/// Uses the identity  ∇u(x) = (3/R) E[ u(x+R ξ) ξ ],  valid when u is harmonic in B(x,R).
/// We estimate u(x+R ξ) by *continuing* WoS from that one-step point.
///
/// Unbiased in expectation; variance ↓ as you increase `grad.boundary_dirs` (and also by
/// averaging multiple calls with different seeds, as usual with MC).
pub fn grad_laplace_dirichlet_wos<D, A, G>(
    domain: &D,
    accel: &A,
    g: &G,
    walk: WosParams,
    grad: GradParams,
    rng: &mut Rng,
    x: Vec3,
) -> Vec3
where
    D: Domain,
    A: ClosestAccel<D>,
    G: BoundaryDirichlet,
{
    let mut eval_u = |d: &D, a: &A, w: WosParams, r: &mut Rng, xp: Vec3| {
        wos_laplace_dirichlet(d, a, g, w, r, xp)
    };
    grad_surface_single_ball(
        domain,
        accel,
        &mut eval_u,
        walk,
        grad.boundary_dirs,
        grad.min_r,
        rng,
        x,
    )
}

/// ### Gradient of the Poisson–Dirichlet solution in 3D via a single-ball estimator.
///
/// We estimate:
/// - surface term by sampling directions ξ and *continuing* Poisson WoS from x+Rξ;
/// - volume term by MC over the ball using either uniform or Green-ball IS.
pub fn grad_poisson_dirichlet_wos<D, A, G, F>(
    domain: &D,
    accel: &A,
    g: &G,
    fsrc: &F,
    walk: WosParams,
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
{
    // Ball at x
    let c = accel.closest(domain, x);
    let R = c.distance.max(grad.min_r);

    // Surface term (needs full Poisson u)
    let mut eval_u = |d: &D, a: &A, w: WosParams, r: &mut Rng, xp: Vec3| {
        wos_poisson_dirichlet(d, a, g, fsrc, w, pois, r, xp)
    };
    let surf = grad_surface_single_ball(
        domain,
        accel,
        &mut eval_u,
        walk,
        grad.boundary_dirs,
        grad.min_r,
        rng,
        x,
    );

    // --- Volume term:  ∫_B ∇_x G f
    let k = grad.interior_samples.max(1);
    let mut vol = Vec3::new(0.0, 0.0, 0.0);

    match grad.sampling {
        InteriorSampling::Uniform => {
            // I ≈ Vol(B) * (1/k) Σ [ ∇_x G(x, Y_i) f(Y_i) ], Y_i ~ Unif(B)
            let volB = ball_volume(R);
            let mut sum = Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..k {
                let y = sample_ball_uniform(rng, x, R);
                let r = (x - y).length().max(grad.min_r);
                let gradG = (x - y) * (-1.0 / (4.0 * PI * r * r * r)); // -(x-y)/4π r^3
                sum += gradG * fsrc.value(y);
            }
            vol = sum * (volB / k as f32);
        }
        InteriorSampling::GreenBall => {
            // p(y) ∝ G(x,y) (with total mass Z = R²/6)
            // ∫ ∇G f = E_p[ (∇G / p) f ] = Z * E_p[ (∇G / G) f ]
            let Z = green_ball_total_mass(R);
            let mut sum = Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..k {
                let y = sample_ball_by_green_pdf(rng, x, R);
                let r = (x - y).length().max(grad.min_r);
                let G = (1.0 / (4.0 * PI)) * (1.0 / r - 1.0 / R).max(1e-12);
                let gradG = (x - y) * (-1.0 / (4.0 * PI * r * r * r));
                // weight = Z * (gradG / G)
                let w = Z / G;
                sum += gradG * (w * fsrc.value(y));
            }
            vol = sum * (1.0 / k as f32);
        }
    }

    // ∇u = surface − volume
    surf - vol
}

/// ### Online mean/variance accumulator (Welford)
///
/// Tracks mean and (unbiased) sample variance of a stream of values.
#[derive(Copy, Clone, Default, Debug)]
pub struct Stats {
    n: u32,
    mean: f32,
    m2: f32,
}
impl Stats {
    #[inline]
    pub fn push(&mut self, x: f32) {
        self.n = self.n.saturating_add(1);
        let n = self.n as f32;
        let delta = x - self.mean;
        self.mean += delta / n;
        self.m2 += delta * (x - self.mean);
    }
    #[inline]
    pub fn mean(&self) -> f32 {
        self.mean
    }
    /// Unbiased sample variance; returns 0 if n<2.
    #[inline]
    pub fn var(&self) -> f32 {
        if self.n > 1 {
            self.m2 / ((self.n - 1) as f32)
        } else {
            0.0
        }
    }
    #[inline]
    pub fn count(&self) -> u32 {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    /// Run many **Poisson** single-path estimates of u(x) and return Welford stats.
    fn poisson_repeat_and_stats<D, A, G, F>(
        trials: u32,
        domain: &D,
        accel: &A,
        g: &G,
        fsrc: &F,
        params: WosParams,
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
            let u = wos_poisson_dirichlet(domain, accel, g, fsrc, params, poisson, &mut rng, x);
            stats.push(u);
        }
        stats
    }

    #[test]
    fn laplacian_dirichlet_smoke() {
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

    /// Constant Dirichlet Poisson in the unit ball:
    ///  - domain: φ(p)=|p|-1 (unit ball),
    ///  - PDE:    -Δ u = 1 in Ω,  u = 0 on ∂Ω,
    ///  - exact:  u(r) = (1 - r^2)/6  in 3D.
    ///
    /// We test at the center (r=0): u(0) = 1/6 ≈ 0.166666...
    #[test]
    fn poisson_unit_ball_constant_source_center() {
        let sphere = SdfDomain::new(|p: Vec3| p.length() - 1.0);
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

        let params = WosParams::new(1e-4, 20_000);
        let poisson = PoissonParams::new(8); // 8 interior samples per step helps variance
        let mut rng = rng::Rng::seed_from(1234);

        // Expect ~1/6 at center
        let u = wos_poisson_dirichlet(
            &sphere,
            &ClosestNaive,
            &g0,
            &fsrc,
            params,
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
        let g = BoundaryDirichletFn::new(|_p| 1.0);
        struct Zero;
        impl SourceTerm for Zero {
            #[inline]
            fn value(&self, _x: Vec3) -> f32 {
                0.0
            }
        }

        let params = WosParams::new(1e-4, 10_000);
        let poisson = PoissonParams::new(1);
        let mut rng1 = rng::Rng::seed_from(7);
        let mut rng2 = rng::Rng::seed_from(7);

        let u_lap = wos_laplace_dirichlet(
            &sphere,
            &ClosestNaive,
            &g,
            params,
            &mut rng1,
            Vec3::new(0.0, 0.0, 0.0),
        );
        let u_pois = wos_poisson_dirichlet(
            &sphere,
            &ClosestNaive,
            &g,
            &Zero,
            params,
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
        let g0 = BoundaryDirichletFn::new(|_p| 0.0);

        struct One;
        impl SourceTerm for One {
            fn value(&self, _x: Vec3) -> f32 {
                1.0
            }
        }
        let fsrc = One;

        let params = WosParams::new(1e-4, 50_000);
        let x0 = Vec3::new(0.2, 0.0, 0.0);

        // Keep interior_samples_per_step small to highlight per-step variance effects
        let uni = PoissonParams::new(1).with_sampling(InteriorSampling::Uniform);
        let grn = PoissonParams::new(1).with_sampling(InteriorSampling::GreenBall);

        let trials = 400; // modest but illustrative
        let s_uni = poisson_repeat_and_stats(
            trials,
            &domain,
            &ClosestNaive,
            &g0,
            &fsrc,
            params,
            uni,
            x0,
            777,
        );
        let s_grn = poisson_repeat_and_stats(
            trials,
            &domain,
            &ClosestNaive,
            &g0,
            &fsrc,
            params,
            grn,
            x0,
            888,
        );

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
        let params = WosParams::new(1e-4, 10_000);
        let poisson = PoissonParams::new(1);
        let mut rng = rng::Rng::seed_from(123);

        // Evaluate at x = 0
        let x = Vec3::new(0.0, 0.0, 0.0);
        let u = wos_poisson_dirichlet(
            &ball,
            &ClosestNaive,
            &g0,
            &fsrc,
            params,
            poisson,
            &mut rng,
            x,
        );

        // Exact Green of unit ball at center:
        // G(0,z) = (1/(4π)) (1/|z| - 1)
        let r = (z - x).length().max(1e-7);
        let exact = (1.0 / (4.0 * core::f32::consts::PI)) * (1.0 / r - 1.0) * q;

        assert!((u - exact).abs() < 1e-4, "u = {u}, exact = {exact}");
    }

    #[test]
    fn grad_laplace_dirichlet_linear_bc_is_constant() {
        let ball = SdfDomain::new(|p: Vec3| p.length() - 1.0);
        // g(p)=p.x ⇒ u(x)=x.x inside the ball (harmonic extension), ∇u=(1,0,0).
        let g_lin = BoundaryDirichletFn::new(|p: Vec3| p.x);
        let walk = WosParams::new(1e-4, 40_000);
        let mut rng = rng::Rng::seed_from(2025);

        // interior point
        let x = Vec3::new(0.2, -0.1, 0.15);

        // A few dozen directions is enough; increase for tighter tolerance.
        let grad_cfg = GradParams::new(512, 1);

        let g_est =
            grad_laplace_dirichlet_wos(&ball, &ClosestNaive, &g_lin, walk, grad_cfg, &mut rng, x);
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
        let g0 = BoundaryDirichletFn::new(|_| 0.0);
        struct One;
        impl SourceTerm for One {
            fn value(&self, _x: Vec3) -> f32 {
                1.0
            }
        }
        let fsrc = One;

        let walk = WosParams::new(1e-4, 60_000);
        // Keep Poisson scalar estimator variance low (used for surface term evaluation):
        let pois = PoissonParams::new(4).with_sampling(InteriorSampling::GreenBall);
        let mut rng = rng::Rng::seed_from(7);

        // interior point
        let x = Vec3::new(0.25, -0.1, 0.2);

        // More samples for the gradient’s surface+volume terms
        let grad_cfg = GradParams::new(256, 64).with_sampling(InteriorSampling::GreenBall);

        let g_est = grad_poisson_dirichlet_wos(
            &ball,
            &ClosestNaive,
            &g0,
            &fsrc,
            walk,
            pois,
            grad_cfg,
            &mut rng,
            x,
        );

        // Analytic: u=(1 - |x|^2)/6 ⇒ ∇u = -(1/3) x
        let g_true = x * (-1.0 / 3.0);
        let err = (g_est - g_true).length();

        assert!(
            err < 0.12,
            "Poisson grad error too large: got {g_est:?}, want {g_true:?}, |err|={err}"
        );
    }
}
