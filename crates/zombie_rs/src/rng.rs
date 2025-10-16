//! Deterministic pseudo-random number generation helpers.

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
