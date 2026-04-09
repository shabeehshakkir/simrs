use std::cmp::Ordering;

/// Optimization direction for simulation objectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

/// A coarse-grained objective evaluation for one design or policy.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEvaluation {
    pub id: String,
    pub objective: f64,
    pub feasible: bool,
}

impl CandidateEvaluation {
    pub fn new(id: impl Into<String>, objective: f64) -> Self {
        Self {
            id: id.into(),
            objective,
            feasible: true,
        }
    }

    pub fn infeasible(id: impl Into<String>, objective: f64) -> Self {
        Self {
            id: id.into(),
            objective,
            feasible: false,
        }
    }
}

/// Return the best feasible candidate under the requested objective sense.
pub fn best_candidate<'a>(
    candidates: &'a [CandidateEvaluation],
    sense: ObjectiveSense,
) -> Option<&'a CandidateEvaluation> {
    candidates
        .iter()
        .filter(|candidate| candidate.feasible)
        .min_by(|a, b| compare_objective(a.objective, b.objective, sense))
}

/// Return a sorted view of feasible candidates from best to worst.
pub fn rank_candidates(
    candidates: &[CandidateEvaluation],
    sense: ObjectiveSense,
) -> Vec<CandidateEvaluation> {
    let mut ranked = candidates
        .iter()
        .filter(|candidate| candidate.feasible)
        .cloned()
        .collect::<Vec<_>>();

    ranked.sort_by(|a, b| compare_objective(a.objective, b.objective, sense));
    ranked
}

fn compare_objective(left: f64, right: f64, sense: ObjectiveSense) -> Ordering {
    match sense {
        ObjectiveSense::Minimize => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
        ObjectiveSense::Maximize => right.partial_cmp(&left).unwrap_or(Ordering::Equal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_candidate_minimization() {
        let candidates = vec![
            CandidateEvaluation::new("a", 10.0),
            CandidateEvaluation::new("b", 7.5),
            CandidateEvaluation::new("c", 8.0),
        ];

        let best = best_candidate(&candidates, ObjectiveSense::Minimize).unwrap();
        assert_eq!(best.id, "b");
    }

    #[test]
    fn best_candidate_skips_infeasible() {
        let candidates = vec![
            CandidateEvaluation::infeasible("a", 1.0),
            CandidateEvaluation::new("b", 2.0),
        ];

        let best = best_candidate(&candidates, ObjectiveSense::Minimize).unwrap();
        assert_eq!(best.id, "b");
    }

    #[test]
    fn rank_candidates_maximization() {
        let ranked = rank_candidates(
            &[
                CandidateEvaluation::new("a", 1.0),
                CandidateEvaluation::new("b", 3.0),
                CandidateEvaluation::new("c", 2.0),
            ],
            ObjectiveSense::Maximize,
        );

        let ids = ranked.iter().map(|candidate| candidate.id.as_str()).collect::<Vec<_>>();
        assert_eq!(ids, vec!["b", "c", "a"]);
    }
}
