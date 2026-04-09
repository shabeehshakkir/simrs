use std::fmt;
use std::ops::{Add, AddAssign, Sub};

/// A simulation time value.
///
/// Wraps `f64` to prevent accidental raw-float comparisons in public APIs
/// and to give time a distinct type in function signatures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimTime(f64);

impl SimTime {
    pub const ZERO: SimTime = SimTime(0.0);
    pub const INFINITY: SimTime = SimTime(f64::INFINITY);

    /// Create a `SimTime` from a raw `f64`.
    ///
    /// # Panics
    /// Panics if `t` is NaN.
    #[inline]
    pub fn from_f64(t: f64) -> Self {
        assert!(!t.is_nan(), "SimTime cannot be NaN");
        SimTime(t)
    }

    /// Return the raw `f64` value.
    #[inline]
    pub fn as_f64(self) -> f64 {
        self.0
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

impl Eq for SimTime {}

impl PartialOrd for SimTime {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SimTime {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // total ordering via f64::total_cmp (stable since Rust 1.62)
        self.0.total_cmp(&other.0)
    }
}

impl Add<f64> for SimTime {
    type Output = SimTime;

    fn add(self, rhs: f64) -> SimTime {
        SimTime::from_f64(self.0 + rhs)
    }
}

impl AddAssign<f64> for SimTime {
    fn add_assign(&mut self, rhs: f64) {
        self.0 += rhs;
    }
}

impl Sub for SimTime {
    type Output = f64;

    fn sub(self, rhs: SimTime) -> f64 {
        self.0 - rhs.0
    }
}

impl fmt::Display for SimTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<f64> for SimTime {
    fn from(t: f64) -> Self {
        SimTime::from_f64(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering() {
        let a = SimTime::from_f64(1.0);
        let b = SimTime::from_f64(2.0);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, SimTime::from_f64(1.0));
    }

    #[test]
    fn arithmetic() {
        let t = SimTime::from_f64(3.0);
        assert_eq!((t + 2.0).as_f64(), 5.0);
        let t2 = SimTime::from_f64(5.0);
        assert_eq!((t2 - t).abs(), 2.0_f64);
    }

    #[test]
    fn zero_and_inf() {
        assert_eq!(SimTime::ZERO.as_f64(), 0.0);
        assert!(!SimTime::ZERO.is_finite() == false);
        assert!(!SimTime::INFINITY.is_finite());
    }

    #[test]
    #[should_panic]
    fn nan_panics() {
        SimTime::from_f64(f64::NAN);
    }
}
