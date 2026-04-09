//! Statistical estimators and inference helpers for simulation output analysis.
//!
//! This module currently focuses on lightweight, dependency-free primitives that
//! are useful across terminating simulation analysis, paired comparisons, and
//! early steady-state workflows.

use std::cmp::Ordering;

/// Summary statistics for an i.i.d. sample.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleSummary {
    pub n: usize,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub std_error: f64,
}

impl SampleSummary {
    pub fn from_slice(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let n = values.len();
        let mean = sample_mean(values)?;
        let variance = sample_variance(values).unwrap_or(0.0);
        let std_dev = variance.sqrt();
        let std_error = if n >= 2 {
            std_dev / (n as f64).sqrt()
        } else {
            0.0
        };

        Some(Self {
            n,
            mean,
            variance,
            std_dev,
            std_error,
        })
    }
}

/// Two-sided confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceInterval {
    pub level: f64,
    pub lower: f64,
    pub upper: f64,
    pub center: f64,
    pub half_width: f64,
}

impl ConfidenceInterval {
    pub fn contains(&self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }
}

/// One-sided confidence bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneSidedConfidenceBound {
    pub level: f64,
    pub bound: f64,
    pub center: f64,
    pub margin: f64,
    pub direction: BoundDirection,
}

impl OneSidedConfidenceBound {
    pub fn contains(&self, value: f64) -> bool {
        match self.direction {
            BoundDirection::Lower => value >= self.bound,
            BoundDirection::Upper => value <= self.bound,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundDirection {
    Lower,
    Upper,
}

/// Relative-precision stopping status for sequential replication analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativePrecisionCheck {
    pub ratio: f64,
    pub threshold: f64,
    pub satisfied: bool,
}

/// Arithmetic mean of a non-empty sample.
pub fn sample_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Unbiased sample variance with denominator n-1.
pub fn sample_variance(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }

    let mean = sample_mean(values)?;
    let sum_sq = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>();

    Some(sum_sq / (values.len() as f64 - 1.0))
}

pub fn sample_std_dev(values: &[f64]) -> Option<f64> {
    sample_variance(values).map(f64::sqrt)
}

pub fn standard_error(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    sample_std_dev(values).map(|sd| sd / (values.len() as f64).sqrt())
}

/// Two-sided t-based confidence interval for the sample mean at a supported
/// confidence level.
///
/// Supported levels are currently 0.90, 0.95, and 0.99.
pub fn mean_confidence_interval(values: &[f64], level: f64) -> Option<ConfidenceInterval> {
    let summary = SampleSummary::from_slice(values)?;
    if summary.n < 2 {
        return None;
    }

    let critical = t_critical_two_sided(level, summary.n - 1)?;
    Some(confidence_interval_from_summary(
        summary.mean,
        summary.std_error,
        level,
        critical,
    ))
}

/// One-sided t-based lower confidence bound for the mean.
pub fn mean_lower_confidence_bound(
    values: &[f64],
    level: f64,
) -> Option<OneSidedConfidenceBound> {
    let summary = SampleSummary::from_slice(values)?;
    if summary.n < 2 {
        return None;
    }

    let critical = t_critical_one_sided(level, summary.n - 1)?;
    Some(one_sided_bound_from_summary(
        summary.mean,
        summary.std_error,
        level,
        critical,
        BoundDirection::Lower,
    ))
}

/// One-sided t-based upper confidence bound for the mean.
pub fn mean_upper_confidence_bound(
    values: &[f64],
    level: f64,
) -> Option<OneSidedConfidenceBound> {
    let summary = SampleSummary::from_slice(values)?;
    if summary.n < 2 {
        return None;
    }

    let critical = t_critical_one_sided(level, summary.n - 1)?;
    Some(one_sided_bound_from_summary(
        summary.mean,
        summary.std_error,
        level,
        critical,
        BoundDirection::Upper,
    ))
}

/// Convenience wrapper for the common 95% two-sided mean CI.
pub fn mean_confidence_interval_95(values: &[f64]) -> Option<ConfidenceInterval> {
    mean_confidence_interval(values, 0.95)
}

pub fn confidence_interval_from_summary(
    center: f64,
    std_error: f64,
    level: f64,
    critical_value: f64,
) -> ConfidenceInterval {
    let half_width = critical_value * std_error;
    ConfidenceInterval {
        level,
        lower: center - half_width,
        upper: center + half_width,
        center,
        half_width,
    }
}

pub fn one_sided_bound_from_summary(
    center: f64,
    std_error: f64,
    level: f64,
    critical_value: f64,
    direction: BoundDirection,
) -> OneSidedConfidenceBound {
    let margin = critical_value * std_error;
    let bound = match direction {
        BoundDirection::Lower => center - margin,
        BoundDirection::Upper => center + margin,
    };

    OneSidedConfidenceBound {
        level,
        bound,
        center,
        margin,
        direction,
    }
}

/// Paired differences `a_i - b_i` for matched-replication comparisons.
pub fn paired_differences(left: &[f64], right: &[f64]) -> Option<Vec<f64>> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }

    Some(
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| l - r)
            .collect(),
    )
}

/// Summary statistics for paired differences.
pub fn paired_difference_summary(left: &[f64], right: &[f64]) -> Option<SampleSummary> {
    let differences = paired_differences(left, right)?;
    SampleSummary::from_slice(&differences)
}

/// Paired-comparison confidence interval based on matched differences.
pub fn paired_mean_confidence_interval(
    left: &[f64],
    right: &[f64],
    level: f64,
) -> Option<ConfidenceInterval> {
    let differences = paired_differences(left, right)?;
    mean_confidence_interval(&differences, level)
}

/// Relative precision ratio `half_width / |center|`.
///
/// Returns `None` if the center is numerically zero.
pub fn relative_precision_ratio(interval: &ConfidenceInterval) -> Option<f64> {
    if interval.center.abs() <= f64::EPSILON {
        return None;
    }
    Some(interval.half_width / interval.center.abs())
}

/// Whether a CI satisfies a target relative precision threshold.
pub fn satisfies_relative_precision(
    interval: &ConfidenceInterval,
    max_relative_half_width: f64,
) -> Option<bool> {
    if max_relative_half_width < 0.0 {
        return None;
    }
    Some(relative_precision_ratio(interval)? <= max_relative_half_width)
}

/// Detailed relative-precision status for a confidence interval.
pub fn relative_precision_check(
    interval: &ConfidenceInterval,
    max_relative_half_width: f64,
) -> Option<RelativePrecisionCheck> {
    if max_relative_half_width < 0.0 {
        return None;
    }
    let ratio = relative_precision_ratio(interval)?;
    Some(RelativePrecisionCheck {
        ratio,
        threshold: max_relative_half_width,
        satisfied: ratio <= max_relative_half_width,
    })
}

/// Non-overlapping batch means from a sequence of replication outputs.
///
/// Any trailing incomplete batch is discarded.
pub fn batch_means(values: &[f64], batch_size: usize) -> Option<Vec<f64>> {
    if batch_size == 0 || values.len() < batch_size {
        return None;
    }

    let full_batches = values.len() / batch_size;
    if full_batches == 0 {
        return None;
    }

    let mut means = Vec::with_capacity(full_batches);
    for batch in values.chunks(batch_size).take(full_batches) {
        means.push(sample_mean(batch)?);
    }
    Some(means)
}

/// Sample summary of non-overlapping batch means.
pub fn nonoverlapping_batch_means_summary(
    values: &[f64],
    batch_size: usize,
) -> Option<SampleSummary> {
    let means = batch_means(values, batch_size)?;
    if means.len() < 2 {
        return None;
    }
    SampleSummary::from_slice(&means)
}

/// Empirical exceedance probability P(X >= threshold).
pub fn exceedance_probability(values: &[f64], threshold: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    let count = values.iter().filter(|value| **value >= threshold).count();
    Some(count as f64 / values.len() as f64)
}

/// Empirical quantile with linear interpolation between order statistics.
pub fn empirical_quantile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    if sorted.len() == 1 {
        return Some(sorted[0]);
    }

    let rank = p * (sorted.len() as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    if lower == upper {
        return Some(sorted[lower]);
    }

    let weight = rank - lower as f64;
    Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
}

/// Binomial-style approximate CI for an exceedance probability.
pub fn exceedance_confidence_interval(
    values: &[f64],
    threshold: f64,
    level: f64,
) -> Option<ConfidenceInterval> {
    let p_hat = exceedance_probability(values, threshold)?;
    if values.len() < 2 {
        return None;
    }
    let z = z_critical_two_sided(level)?;
    let se = (p_hat * (1.0 - p_hat) / values.len() as f64).sqrt();
    let ci = confidence_interval_from_summary(p_hat, se, level, z);
    Some(ConfidenceInterval {
        lower: ci.lower.clamp(0.0, 1.0),
        upper: ci.upper.clamp(0.0, 1.0),
        ..ci
    })
}

/// Time-weighted average for a piecewise-constant process over [0, horizon].
///
/// `samples` is a sequence of `(time, value)` pairs that indicate the value in
/// force starting at `time` until the next sample time. The first sample must
/// start at time 0 and sample times must be nondecreasing.
pub fn time_weighted_mean(samples: &[(f64, f64)], horizon: f64) -> Option<f64> {
    if samples.is_empty() || !horizon.is_finite() || horizon <= 0.0 {
        return None;
    }
    if samples[0].0 != 0.0 {
        return None;
    }

    let mut area = 0.0;

    for window in samples.windows(2) {
        let (start, value) = window[0];
        let (end, _) = window[1];
        if end < start {
            return None;
        }
        area += (end - start) * value;
    }

    let (last_time, last_value) = *samples.last()?;
    if last_time > horizon {
        return None;
    }
    area += (horizon - last_time) * last_value;

    Some(area / horizon)
}

fn t_critical_two_sided(level: f64, df: usize) -> Option<f64> {
    if df == 0 {
        return None;
    }

    let table = match level_key(level)? {
        90 => &T_CRITICAL_90_TWO_SIDED[..],
        95 => &T_CRITICAL_95_TWO_SIDED[..],
        99 => &T_CRITICAL_99_TWO_SIDED[..],
        _ => return None,
    };

    Some(if df <= 30 {
        table[df - 1]
    } else {
        z_critical_two_sided(level)?
    })
}

fn t_critical_one_sided(level: f64, df: usize) -> Option<f64> {
    if df == 0 {
        return None;
    }

    let table = match level_key(level)? {
        90 => &T_CRITICAL_90_ONE_SIDED[..],
        95 => &T_CRITICAL_95_ONE_SIDED[..],
        99 => &T_CRITICAL_99_ONE_SIDED[..],
        _ => return None,
    };

    Some(if df <= 30 {
        table[df - 1]
    } else {
        z_critical_one_sided(level)?
    })
}

fn z_critical_two_sided(level: f64) -> Option<f64> {
    match level_key(level)? {
        90 => Some(1.644_853_626_951_472_2),
        95 => Some(1.959_963_984_540_054),
        99 => Some(2.575_829_303_548_900_4),
        _ => None,
    }
}

fn z_critical_one_sided(level: f64) -> Option<f64> {
    match level_key(level)? {
        90 => Some(1.281_551_565_544_600_4),
        95 => Some(1.644_853_626_951_472_2),
        99 => Some(2.326_347_874_040_840_8),
        _ => None,
    }
}

fn level_key(level: f64) -> Option<u32> {
    if !level.is_finite() || !(0.0..1.0).contains(&level) {
        return None;
    }
    Some((level * 100.0).round() as u32)
}

const T_CRITICAL_90_TWO_SIDED: [f64; 30] = [
    6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771,
    1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706,
    1.703, 1.701, 1.699, 1.697,
];
const T_CRITICAL_95_TWO_SIDED: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];
const T_CRITICAL_99_TWO_SIDED: [f64; 30] = [
    63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169, 3.106, 3.055, 3.012,
    2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845, 2.831, 2.819, 2.807, 2.797, 2.787, 2.779,
    2.771, 2.763, 2.756, 2.750,
];
const T_CRITICAL_90_ONE_SIDED: [f64; 30] = [
    3.078, 1.886, 1.638, 1.533, 1.476, 1.440, 1.415, 1.397, 1.383, 1.372, 1.363, 1.356, 1.350,
    1.345, 1.341, 1.337, 1.333, 1.330, 1.328, 1.325, 1.323, 1.321, 1.319, 1.318, 1.316, 1.315,
    1.314, 1.313, 1.311, 1.310,
];
const T_CRITICAL_95_ONE_SIDED: [f64; 30] = [
    6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771,
    1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706,
    1.703, 1.701, 1.699, 1.697,
];
const T_CRITICAL_99_ONE_SIDED: [f64; 30] = [
    31.821, 6.965, 4.541, 3.747, 3.365, 3.143, 2.998, 2.896, 2.821, 2.764, 2.718, 2.681, 2.650,
    2.624, 2.602, 2.583, 2.567, 2.552, 2.539, 2.528, 2.518, 2.508, 2.500, 2.492, 2.485, 2.479,
    2.473, 2.467, 2.462, 2.457,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_summary_basic() {
        let summary = SampleSummary::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(summary.n, 3);
        assert!((summary.mean - 2.0).abs() < 1e-12);
        assert!((summary.variance - 1.0).abs() < 1e-12);
    }

    #[test]
    fn standard_error_none_for_singleton() {
        assert_eq!(standard_error(&[1.0]), None);
    }

    #[test]
    fn generic_confidence_interval_matches_95_helper() {
        let a = mean_confidence_interval(&[1.0, 2.0, 3.0, 4.0], 0.95).unwrap();
        let b = mean_confidence_interval_95(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn one_sided_bounds_bracket_center_in_expected_direction() {
        let lower = mean_lower_confidence_bound(&[1.0, 2.0, 3.0, 4.0], 0.95).unwrap();
        let upper = mean_upper_confidence_bound(&[1.0, 2.0, 3.0, 4.0], 0.95).unwrap();
        assert!(lower.bound < lower.center);
        assert!(upper.bound > upper.center);
        assert!(lower.contains(lower.center));
        assert!(upper.contains(upper.center));
    }

    #[test]
    fn confidence_interval_has_mean_as_center() {
        let ci = mean_confidence_interval_95(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((ci.center - 2.5).abs() < 1e-12);
        assert!(ci.lower < ci.center);
        assert!(ci.upper > ci.center);
        assert!(ci.contains(2.5));
    }

    #[test]
    fn paired_difference_computation() {
        let diffs = paired_differences(&[5.0, 4.0], &[2.0, 1.5]).unwrap();
        assert_eq!(diffs, vec![3.0, 2.5]);
    }

    #[test]
    fn paired_summary_and_ci_work() {
        let summary = paired_difference_summary(&[5.0, 6.0, 7.0], &[3.0, 4.0, 6.0]).unwrap();
        assert!((summary.mean - 5.0 / 3.0).abs() < 1e-12);

        let ci = paired_mean_confidence_interval(&[5.0, 6.0, 7.0], &[3.0, 4.0, 6.0], 0.95).unwrap();
        assert!(ci.lower < ci.center);
        assert!(ci.upper > ci.center);
    }

    #[test]
    fn relative_precision_checks_behave_sensibly() {
        let ci = confidence_interval_from_summary(10.0, 0.5, 0.95, 2.0);
        let ratio = relative_precision_ratio(&ci).unwrap();
        assert!((ratio - 0.1).abs() < 1e-12);
        assert_eq!(satisfies_relative_precision(&ci, 0.11), Some(true));
        assert_eq!(satisfies_relative_precision(&ci, 0.09), Some(false));

        let check = relative_precision_check(&ci, 0.11).unwrap();
        assert!(check.satisfied);
        assert_eq!(check.threshold, 0.11);
    }

    #[test]
    fn batch_means_builds_expected_batches() {
        let means = batch_means(&[1.0, 3.0, 5.0, 7.0, 10.0, 14.0], 2).unwrap();
        assert_eq!(means, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn batch_means_summary_requires_multiple_batches() {
        let summary = nonoverlapping_batch_means_summary(&[1.0, 3.0, 5.0, 7.0], 2).unwrap();
        assert_eq!(summary.n, 2);
        assert!((summary.mean - 4.0).abs() < 1e-12);

        assert_eq!(nonoverlapping_batch_means_summary(&[1.0, 2.0], 2), None);
    }

    #[test]
    fn exceedance_and_quantile_estimators_work() {
        let p = exceedance_probability(&[1.0, 2.0, 5.0, 7.0], 5.0).unwrap();
        assert!((p - 0.5).abs() < 1e-12);

        let q = empirical_quantile(&[1.0, 5.0, 2.0, 3.0], 0.5).unwrap();
        assert!((q - 2.5).abs() < 1e-12);
    }

    #[test]
    fn exceedance_confidence_interval_is_bounded() {
        let ci = exceedance_confidence_interval(&[1.0, 2.0, 5.0, 7.0, 9.0], 5.0, 0.95).unwrap();
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0);
        assert!(ci.contains(ci.center));
    }

    #[test]
    fn unsupported_confidence_level_returns_none() {
        assert_eq!(mean_confidence_interval(&[1.0, 2.0], 0.92), None);
        assert_eq!(mean_lower_confidence_bound(&[1.0, 2.0], 0.92), None);
    }

    #[test]
    fn time_weighted_mean_for_piecewise_constant_trace() {
        let samples = vec![(0.0, 1.0), (2.0, 3.0), (5.0, 2.0)];
        let mean = time_weighted_mean(&samples, 10.0).unwrap();
        let expected = (2.0 * 1.0 + 3.0 * 3.0 + 5.0 * 2.0) / 10.0;
        assert!((mean - expected).abs() < 1e-12);
    }

    #[test]
    fn time_weighted_mean_rejects_bad_traces() {
        assert_eq!(time_weighted_mean(&[(1.0, 2.0)], 3.0), None);
        assert_eq!(time_weighted_mean(&[(0.0, 2.0), (1.0, 3.0)], 0.0), None);
        assert_eq!(
            time_weighted_mean(&[(0.0, 2.0), (2.0, 3.0), (1.0, 4.0)], 5.0),
            None
        );
    }
}
