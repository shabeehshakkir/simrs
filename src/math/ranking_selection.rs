use crate::math::optimization::{best_candidate, CandidateEvaluation, ObjectiveSense};

/// Basic indifference-zone configuration for ranking-and-selection procedures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndifferenceZoneConfig {
    pub delta: f64,
    pub alpha: f64,
}

impl IndifferenceZoneConfig {
    pub fn new(delta: f64, alpha: f64) -> Option<Self> {
        if delta <= 0.0 || !(0.0..1.0).contains(&alpha) {
            return None;
        }
        Some(Self { delta, alpha })
    }
}

/// Per-alternative summary from an initial sampling stage.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionAlternative {
    pub id: String,
    pub mean: f64,
    pub sample_variance: f64,
    pub samples: usize,
    pub feasible: bool,
}

impl SelectionAlternative {
    pub fn new(
        id: impl Into<String>,
        mean: f64,
        sample_variance: f64,
        samples: usize,
    ) -> Option<Self> {
        if sample_variance < 0.0 || samples < 2 {
            return None;
        }
        Some(Self {
            id: id.into(),
            mean,
            sample_variance,
            samples,
            feasible: true,
        })
    }

    pub fn to_candidate(&self) -> CandidateEvaluation {
        CandidateEvaluation {
            id: self.id.clone(),
            objective: self.mean,
            feasible: self.feasible,
        }
    }

    pub fn standard_error(&self) -> f64 {
        (self.sample_variance / self.samples as f64).sqrt()
    }
}

/// High-level PCS-oriented summary for an R&S run.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionDecision {
    pub selected_id: String,
    pub contenders: Vec<String>,
    pub eliminated: Vec<String>,
    pub total_samples: usize,
    pub estimated_pcs_lower_bound: Option<f64>,
}

/// Per-alternative budget recommendation for the next sampling wave.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetAllocation {
    pub id: String,
    pub weight: f64,
    pub additional_samples: usize,
}

/// Result of a simple elimination pass.
#[derive(Debug, Clone, PartialEq)]
pub struct EliminationResult {
    pub survivors: Vec<SelectionAlternative>,
    pub eliminated: Vec<SelectionAlternative>,
}

/// Return the current best alternative by sample mean.
pub fn select_best_by_sample_mean<'a>(
    alternatives: &'a [SelectionAlternative],
    sense: ObjectiveSense,
) -> Option<&'a SelectionAlternative> {
    let candidates = alternatives
        .iter()
        .map(SelectionAlternative::to_candidate)
        .collect::<Vec<_>>();
    let best = best_candidate(&candidates, sense)?;
    alternatives.iter().find(|alt| alt.id == best.id)
}

/// Rinott-style total sample size for one alternative after stage 1.
///
/// This uses the classical structure
/// n_i = max(n0, ceil((h^2 * s_i^2) / delta^2))
///
/// where `h` is a supplied Rinott constant, `s_i^2` is the stage-1 sample
/// variance estimate, and `delta` is the indifference-zone parameter.
pub fn rinott_required_total_samples(
    stage1_samples: usize,
    sample_variance: f64,
    rinott_constant: f64,
    delta: f64,
) -> Option<usize> {
    if stage1_samples < 2 || sample_variance < 0.0 || rinott_constant <= 0.0 || delta <= 0.0 {
        return None;
    }

    let required =
        ((rinott_constant * rinott_constant * sample_variance) / (delta * delta)).ceil() as usize;
    Some(required.max(stage1_samples))
}

/// Additional samples needed after stage 1 under a Rinott-style rule.
pub fn rinott_additional_samples(
    stage1_samples: usize,
    sample_variance: f64,
    rinott_constant: f64,
    delta: f64,
) -> Option<usize> {
    let total =
        rinott_required_total_samples(stage1_samples, sample_variance, rinott_constant, delta)?;
    Some(total.saturating_sub(stage1_samples))
}

/// Return alternatives whose sample means lie within the indifference-zone band
/// of the current best alternative.
pub fn contenders_within_indifference_zone(
    alternatives: &[SelectionAlternative],
    sense: ObjectiveSense,
    config: IndifferenceZoneConfig,
) -> Vec<SelectionAlternative> {
    let Some(best) = select_best_by_sample_mean(alternatives, sense) else {
        return Vec::new();
    };

    alternatives
        .iter()
        .filter(|alt| alt.feasible)
        .filter(|alt| match sense {
            ObjectiveSense::Minimize => alt.mean <= best.mean + config.delta,
            ObjectiveSense::Maximize => alt.mean >= best.mean - config.delta,
        })
        .cloned()
        .collect()
}

/// Eliminate alternatives that are farther than `delta` from the current best.
pub fn eliminate_by_indifference_zone(
    alternatives: &[SelectionAlternative],
    sense: ObjectiveSense,
    config: IndifferenceZoneConfig,
) -> EliminationResult {
    let survivors = contenders_within_indifference_zone(alternatives, sense, config);
    let survivor_ids = survivors.iter().map(|alt| alt.id.as_str()).collect::<Vec<_>>();

    let eliminated = alternatives
        .iter()
        .filter(|alt| alt.feasible)
        .filter(|alt| !survivor_ids.contains(&alt.id.as_str()))
        .cloned()
        .collect::<Vec<_>>();

    EliminationResult {
        survivors,
        eliminated,
    }
}

/// Allocate an integer additional sampling budget in proportion to sample
/// standard errors. This is a simple variance-aware heuristic, not full OCBA.
pub fn allocate_additional_budget_by_standard_error(
    alternatives: &[SelectionAlternative],
    total_additional_samples: usize,
) -> Vec<BudgetAllocation> {
    let feasible = alternatives
        .iter()
        .filter(|alt| alt.feasible)
        .cloned()
        .collect::<Vec<_>>();

    if feasible.is_empty() || total_additional_samples == 0 {
        return feasible
            .into_iter()
            .map(|alt| BudgetAllocation {
                id: alt.id,
                weight: 0.0,
                additional_samples: 0,
            })
            .collect();
    }

    let total_weight = feasible
        .iter()
        .map(SelectionAlternative::standard_error)
        .sum::<f64>();

    if total_weight <= 0.0 {
        let base = total_additional_samples / feasible.len();
        let remainder = total_additional_samples % feasible.len();

        return feasible
            .into_iter()
            .enumerate()
            .map(|(idx, alt)| BudgetAllocation {
                id: alt.id,
                weight: 1.0 / alternatives.len() as f64,
                additional_samples: base + usize::from(idx < remainder),
            })
            .collect();
    }

    let mut allocations = feasible
        .iter()
        .map(|alt| {
            let weight = alt.standard_error() / total_weight;
            let raw = weight * total_additional_samples as f64;
            BudgetAllocation {
                id: alt.id.clone(),
                weight,
                additional_samples: raw.floor() as usize,
            }
        })
        .collect::<Vec<_>>();

    let allocated = allocations
        .iter()
        .map(|allocation| allocation.additional_samples)
        .sum::<usize>();
    let mut remaining = total_additional_samples.saturating_sub(allocated);

    allocations.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));

    let mut i = 0;
    while remaining > 0 && !allocations.is_empty() {
        let len = allocations.len();
        let idx = i % len;
        allocations[idx].additional_samples += 1;
        remaining -= 1;
        i += 1;
    }

    allocations.sort_by(|a, b| a.id.cmp(&b.id));
    allocations
}

/// Bonferroni-style lower bound on PCS from pairwise one-sided confidence events.
pub fn bonferroni_pcs_lower_bound(k: usize, pairwise_error_probability: f64) -> Option<f64> {
    if k < 2 || !(0.0..=1.0).contains(&pairwise_error_probability) {
        return None;
    }

    let pairs = k.saturating_sub(1) as f64;
    Some((1.0 - pairs * pairwise_error_probability).clamp(0.0, 1.0))
}

/// Build a coarse PCS-oriented decision summary from the current alternatives.
pub fn summarize_selection(
    alternatives: &[SelectionAlternative],
    sense: ObjectiveSense,
    pcs_lower_bound: Option<f64>,
) -> Option<SelectionDecision> {
    let best = select_best_by_sample_mean(alternatives, sense)?;
    let contenders = alternatives
        .iter()
        .filter(|alt| alt.feasible)
        .map(|alt| alt.id.clone())
        .collect::<Vec<_>>();
    let eliminated = alternatives
        .iter()
        .filter(|alt| !alt.feasible)
        .map(|alt| alt.id.clone())
        .collect::<Vec<_>>();
    let total_samples = alternatives.iter().map(|alt| alt.samples).sum();

    Some(SelectionDecision {
        selected_id: best.id.clone(),
        contenders,
        eliminated,
        total_samples,
        estimated_pcs_lower_bound: pcs_lower_bound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iz_config_validation() {
        assert!(IndifferenceZoneConfig::new(0.1, 0.05).is_some());
        assert!(IndifferenceZoneConfig::new(0.0, 0.05).is_none());
        assert!(IndifferenceZoneConfig::new(0.1, 1.0).is_none());
    }

    #[test]
    fn best_selection_uses_sample_mean() {
        let alternatives = vec![
            SelectionAlternative::new("a", 10.0, 4.0, 10).unwrap(),
            SelectionAlternative::new("b", 7.0, 2.0, 10).unwrap(),
            SelectionAlternative::new("c", 9.0, 1.0, 10).unwrap(),
        ];

        let best = select_best_by_sample_mean(&alternatives, ObjectiveSense::Minimize).unwrap();
        assert_eq!(best.id, "b");
    }

    #[test]
    fn rinott_sample_formula_matches_closed_form() {
        let total = rinott_required_total_samples(10, 4.0, 2.5, 1.0).unwrap();
        assert_eq!(total, 25);

        let extra = rinott_additional_samples(10, 4.0, 2.5, 1.0).unwrap();
        assert_eq!(extra, 15);
    }

    #[test]
    fn contenders_keep_close_alternatives() {
        let alternatives = vec![
            SelectionAlternative::new("a", 10.0, 1.0, 8).unwrap(),
            SelectionAlternative::new("b", 10.4, 1.0, 8).unwrap(),
            SelectionAlternative::new("c", 11.0, 1.0, 8).unwrap(),
        ];
        let config = IndifferenceZoneConfig::new(0.5, 0.05).unwrap();

        let contenders =
            contenders_within_indifference_zone(&alternatives, ObjectiveSense::Minimize, config);
        let ids = contenders
            .iter()
            .map(|alt| alt.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn elimination_rule_splits_survivors_and_losers() {
        let alternatives = vec![
            SelectionAlternative::new("a", 10.0, 1.0, 8).unwrap(),
            SelectionAlternative::new("b", 10.4, 1.0, 8).unwrap(),
            SelectionAlternative::new("c", 11.0, 1.0, 8).unwrap(),
        ];
        let config = IndifferenceZoneConfig::new(0.5, 0.05).unwrap();

        let result = eliminate_by_indifference_zone(&alternatives, ObjectiveSense::Minimize, config);
        assert_eq!(result.survivors.len(), 2);
        assert_eq!(result.eliminated.len(), 1);
        assert_eq!(result.eliminated[0].id, "c");
    }

    #[test]
    fn budget_allocation_prefers_higher_standard_error() {
        let alternatives = vec![
            SelectionAlternative::new("a", 10.0, 1.0, 10).unwrap(),
            SelectionAlternative::new("b", 10.0, 9.0, 10).unwrap(),
            SelectionAlternative::new("c", 10.0, 4.0, 10).unwrap(),
        ];

        let allocations = allocate_additional_budget_by_standard_error(&alternatives, 12);
        let total = allocations.iter().map(|item| item.additional_samples).sum::<usize>();
        assert_eq!(total, 12);

        let a = allocations.iter().find(|x| x.id == "a").unwrap().additional_samples;
        let b = allocations.iter().find(|x| x.id == "b").unwrap().additional_samples;
        let c = allocations.iter().find(|x| x.id == "c").unwrap().additional_samples;
        assert!(b >= c);
        assert!(c >= a);
    }

    #[test]
    fn pcs_lower_bound_behaves_sensibly() {
        let pcs = bonferroni_pcs_lower_bound(4, 0.02).unwrap();
        assert!((pcs - 0.94).abs() < 1e-12);
    }

    #[test]
    fn selection_summary_collects_ids_and_samples() {
        let mut alternatives = vec![
            SelectionAlternative::new("a", 8.0, 1.0, 10).unwrap(),
            SelectionAlternative::new("b", 7.0, 1.0, 12).unwrap(),
        ];
        alternatives[0].feasible = false;

        let summary = summarize_selection(&alternatives, ObjectiveSense::Minimize, Some(0.9)).unwrap();
        assert_eq!(summary.selected_id, "b");
        assert_eq!(summary.total_samples, 22);
        assert_eq!(summary.eliminated, vec!["a"]);
        assert_eq!(summary.estimated_pcs_lower_bound, Some(0.9));
    }
}
