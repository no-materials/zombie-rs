//! Boundary condition abstractions.

use crate::math::Vec3;

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
