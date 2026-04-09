use std::cmp::Ordering;

/// Empirical quantile with linear interpolation between order statistics.
pub fn quantile(values: &[f64], p: f64) -> Option<f64> {
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

/// Empirical CVaR (expected shortfall) of the upper tail.
pub fn cvar_upper(values: &[f64], alpha: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..1.0).contains(&alpha) {
        return None;
    }

    let threshold = quantile(values, alpha)?;
    let mut tail_count = 0usize;
    let tail_sum = values
        .iter()
        .filter(|value| **value >= threshold)
        .inspect(|_| tail_count += 1)
        .sum::<f64>();

    if tail_count == 0 {
        return None;
    }

    Some(tail_sum / tail_count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_quantile() {
        let q = quantile(&[1.0, 5.0, 2.0, 3.0], 0.5).unwrap();
        assert!((q - 2.5).abs() < 1e-12);
    }

    #[test]
    fn upper_tail_cvar() {
        let cvar = cvar_upper(&[1.0, 2.0, 3.0, 10.0], 0.75).unwrap();
        assert!((cvar - 10.0).abs() < 1e-12);
    }
}
