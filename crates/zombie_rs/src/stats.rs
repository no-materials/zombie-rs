//! Streaming statistics helpers.

/// Online mean/variance accumulator (Welford)
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
