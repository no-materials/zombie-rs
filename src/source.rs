//! Source term interfaces for Poisson-style estimators.

use crate::math::Vec3;

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
