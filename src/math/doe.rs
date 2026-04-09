/// Generate a full-factorial design from factor level sets.
///
/// Each entry in `levels` is the ordered set of levels for one factor.
pub fn full_factorial(levels: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if levels.is_empty() {
        return vec![Vec::new()];
    }
    if levels.iter().any(|factor_levels| factor_levels.is_empty()) {
        return Vec::new();
    }

    let mut design = vec![Vec::new()];

    for factor_levels in levels {
        let mut next_design = Vec::with_capacity(design.len() * factor_levels.len());
        for row in &design {
            for &level in factor_levels {
                let mut next_row = row.clone();
                next_row.push(level);
                next_design.push(next_row);
            }
        }
        design = next_design;
    }

    design
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_two_by_two_design() {
        let design = full_factorial(&[vec![0.0, 1.0], vec![10.0, 20.0]]);
        assert_eq!(
            design,
            vec![
                vec![0.0, 10.0],
                vec![0.0, 20.0],
                vec![1.0, 10.0],
                vec![1.0, 20.0],
            ]
        );
    }

    #[test]
    fn empty_factor_list_yields_single_empty_run() {
        assert_eq!(full_factorial(&[]), vec![Vec::<f64>::new()]);
    }
}
