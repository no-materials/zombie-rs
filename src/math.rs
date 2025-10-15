//! Minimal 3D math utilities tailored for `no_std` Monte Carlo geometry processing.

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use libm::sqrtf;

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
