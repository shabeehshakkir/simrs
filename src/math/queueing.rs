//! Closed-form and steady-state queueing formulas used by OR workflows.
//!
//! This module currently focuses on lightweight formulas for classic Markovian
//! queues, with explicit guards for invalid or undefined parameter regions.

/// Common steady-state metrics for single-class queueing models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueueMetrics {
    pub mean_number_in_queue: f64,
    pub mean_number_in_system: f64,
    pub mean_waiting_time_in_queue: f64,
    pub mean_time_in_system: f64,
}

/// Steady-state metrics for M/M/c-style service systems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiServerQueueMetrics {
    pub probability_zero: f64,
    pub wait_probability: f64,
    pub immediate_service_probability: f64,
    pub mean_number_in_queue: f64,
    pub mean_number_in_system: f64,
    pub mean_waiting_time_in_queue: f64,
    pub mean_time_in_system: f64,
}

/// Steady-state metrics for finite-capacity M/M/1/K systems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteCapacityQueueMetrics {
    pub probability_zero: f64,
    pub blocking_probability: f64,
    pub effective_arrival_rate: f64,
    pub mean_number_in_queue: f64,
    pub mean_number_in_system: f64,
    pub mean_waiting_time_in_queue: f64,
    pub mean_time_in_system: f64,
}

/// Steady-state metrics for Erlang A (M/M/c+M) systems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErlangAMetrics {
    pub probability_zero: f64,
    pub delay_probability: f64,
    pub service_probability: f64,
    pub abandonment_probability: f64,
    pub mean_number_in_queue: f64,
    pub mean_number_in_system: f64,
    pub mean_waiting_time_in_queue: f64,
    pub mean_time_in_system: f64,
}

/// Utilization for a single-server queueing station.
///
/// Returns `None` if `service_rate <= 0`.
pub fn utilization(arrival_rate: f64, service_rate: f64) -> Option<f64> {
    if service_rate <= 0.0 || arrival_rate < 0.0 || !arrival_rate.is_finite() || !service_rate.is_finite() {
        return None;
    }
    Some(arrival_rate / service_rate)
}

/// Utilization for a c-server queueing station.
///
/// Returns `None` if `servers == 0` or `service_rate <= 0`.
pub fn multi_server_utilization(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
) -> Option<f64> {
    if servers == 0
        || service_rate <= 0.0
        || arrival_rate < 0.0
        || !arrival_rate.is_finite()
        || !service_rate.is_finite()
    {
        return None;
    }
    Some(arrival_rate / (servers as f64 * service_rate))
}

/// Little's Law: L = lambda * W.
pub fn littles_l(arrival_rate: f64, mean_time_in_system: f64) -> f64 {
    arrival_rate * mean_time_in_system
}

/// Little's Law rearranged: W = L / lambda.
///
/// Returns `None` if `arrival_rate <= 0`.
pub fn littles_w(mean_number_in_system: f64, arrival_rate: f64) -> Option<f64> {
    if arrival_rate <= 0.0 {
        return None;
    }
    Some(mean_number_in_system / arrival_rate)
}

/// M/M/1 mean number in queue: Lq = rho^2 / (1 - rho).
pub fn mm1_mean_number_in_queue(arrival_rate: f64, service_rate: f64) -> Option<f64> {
    let rho = utilization(arrival_rate, service_rate)?;
    if !(0.0..1.0).contains(&rho) {
        return None;
    }
    Some((rho * rho) / (1.0 - rho))
}

/// M/M/1 mean number in system: L = rho / (1 - rho).
pub fn mm1_mean_number_in_system(arrival_rate: f64, service_rate: f64) -> Option<f64> {
    let rho = utilization(arrival_rate, service_rate)?;
    if !(0.0..1.0).contains(&rho) {
        return None;
    }
    Some(rho / (1.0 - rho))
}

/// M/M/1 mean waiting time in queue: Wq = rho / (mu - lambda).
pub fn mm1_mean_waiting_time_in_queue(arrival_rate: f64, service_rate: f64) -> Option<f64> {
    let rho = utilization(arrival_rate, service_rate)?;
    if !(0.0..1.0).contains(&rho) {
        return None;
    }
    Some(rho / (service_rate - arrival_rate))
}

/// M/M/1 mean time in system: W = 1 / (mu - lambda).
pub fn mm1_mean_time_in_system(arrival_rate: f64, service_rate: f64) -> Option<f64> {
    if service_rate <= arrival_rate || arrival_rate < 0.0 || service_rate <= 0.0 {
        return None;
    }
    Some(1.0 / (service_rate - arrival_rate))
}

/// Erlang B blocking probability for M/M/c/c.
pub fn erlang_b(offered_load: f64, servers: usize) -> Option<f64> {
    if servers == 0 || offered_load < 0.0 || !offered_load.is_finite() {
        return None;
    }

    let mut b = 1.0;
    for n in 1..=servers {
        b = (offered_load * b) / (n as f64 + offered_load * b);
    }
    Some(b)
}

/// Probability that an M/M/c system is empty.
pub fn mmc_probability_zero(arrival_rate: f64, service_rate: f64, servers: usize) -> Option<f64> {
    if servers == 0 || service_rate <= 0.0 || arrival_rate < 0.0 {
        return None;
    }

    let a = arrival_rate / service_rate;
    let rho = multi_server_utilization(arrival_rate, service_rate, servers)?;
    if rho >= 1.0 {
        return None;
    }

    let mut sum = 0.0;
    for n in 0..servers {
        sum += a.powi(n as i32) / factorial(n);
    }

    let tail = a.powi(servers as i32) / (factorial(servers) * (1.0 - rho));
    Some(1.0 / (sum + tail))
}

/// Erlang C delay probability for M/M/c.
pub fn erlang_c(arrival_rate: f64, service_rate: f64, servers: usize) -> Option<f64> {
    let p0 = mmc_probability_zero(arrival_rate, service_rate, servers)?;
    let a = arrival_rate / service_rate;
    let rho = multi_server_utilization(arrival_rate, service_rate, servers)?;

    let numerator = a.powi(servers as i32) / (factorial(servers) * (1.0 - rho));
    Some(numerator * p0)
}

/// M/M/c mean number in queue.
pub fn mmc_mean_number_in_queue(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
) -> Option<f64> {
    let wait_probability = erlang_c(arrival_rate, service_rate, servers)?;
    let rho = multi_server_utilization(arrival_rate, service_rate, servers)?;
    if rho >= 1.0 {
        return None;
    }
    Some(wait_probability * rho / (1.0 - rho))
}

/// M/M/c mean waiting time in queue.
pub fn mmc_mean_waiting_time_in_queue(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
) -> Option<f64> {
    let lq = mmc_mean_number_in_queue(arrival_rate, service_rate, servers)?;
    littles_w(lq, arrival_rate)
}

/// M/M/c mean time in system.
pub fn mmc_mean_time_in_system(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
) -> Option<f64> {
    let wq = mmc_mean_waiting_time_in_queue(arrival_rate, service_rate, servers)?;
    Some(wq + 1.0 / service_rate)
}

/// M/M/c mean number in system.
pub fn mmc_mean_number_in_system(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
) -> Option<f64> {
    let w = mmc_mean_time_in_system(arrival_rate, service_rate, servers)?;
    Some(littles_l(arrival_rate, w))
}

/// Alias for the Erlang C probability of delay in M/M/c.
pub fn mmc_wait_probability(arrival_rate: f64, service_rate: f64, servers: usize) -> Option<f64> {
    erlang_c(arrival_rate, service_rate, servers)
}

/// Probability that the queue wait exceeds a target `t` in M/M/c.
///
/// Conditional on waiting, the M/M/c queue wait is exponential with rate
/// `c * mu - lambda`, so
/// `P(W_q > t) = ErlangC * exp(-(c*mu - lambda)t)` for `t >= 0`.
pub fn mmc_wait_exceeds_probability(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    target_wait: f64,
) -> Option<f64> {
    if target_wait < 0.0 || !target_wait.is_finite() {
        return None;
    }

    let pwait = erlang_c(arrival_rate, service_rate, servers)?;
    let decay_rate = servers as f64 * service_rate - arrival_rate;
    if decay_rate <= 0.0 {
        return None;
    }

    Some(pwait * (-decay_rate * target_wait).exp())
}

/// Probability that the queue wait is at most a target `t` in M/M/c.
pub fn mmc_service_level(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    target_wait: f64,
) -> Option<f64> {
    let exceed = mmc_wait_exceeds_probability(arrival_rate, service_rate, servers, target_wait)?;
    Some(1.0 - exceed)
}

/// Probability that service begins immediately in M/M/c.
pub fn mmc_immediate_service_probability(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
) -> Option<f64> {
    let pwait = mmc_wait_probability(arrival_rate, service_rate, servers)?;
    Some(1.0 - pwait)
}

/// Probability that total time in system is at most `t` in M/M/c.
///
/// This uses the standard decomposition:
/// - with probability `1 - ErlangC`, service starts immediately and system time
///   is exponential with rate `mu`
/// - with probability `ErlangC`, queue wait is exponential with rate
///   `c*mu - lambda`, followed by exponential service with rate `mu`
pub fn mmc_system_time_cdf(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    target_time: f64,
) -> Option<f64> {
    if target_time < 0.0 || !target_time.is_finite() {
        return None;
    }

    let pwait = mmc_wait_probability(arrival_rate, service_rate, servers)?;
    let alpha = servers as f64 * service_rate - arrival_rate;
    let beta = service_rate;
    if alpha <= 0.0 || beta <= 0.0 {
        return None;
    }

    let immediate_cdf = 1.0 - (-beta * target_time).exp();
    let delayed_cdf = if (alpha - beta).abs() <= 1e-12 {
        1.0 - (-alpha * target_time).exp() * (1.0 + alpha * target_time)
    } else {
        1.0 - (beta * (-alpha * target_time).exp() - alpha * (-beta * target_time).exp())
            / (beta - alpha)
    };

    Some((1.0 - pwait) * immediate_cdf + pwait * delayed_cdf)
}

/// Probability that total time in system exceeds `t` in M/M/c.
pub fn mmc_system_time_exceeds_probability(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    target_time: f64,
) -> Option<f64> {
    let cdf = mmc_system_time_cdf(arrival_rate, service_rate, servers, target_time)?;
    Some(1.0 - cdf)
}

/// Probability that an M/M/1/K system is empty.
pub fn mm1k_probability_zero(arrival_rate: f64, service_rate: f64, capacity: usize) -> Option<f64> {
    let rho = utilization(arrival_rate, service_rate)?;
    if capacity == 0 {
        return Some(1.0);
    }

    if (rho - 1.0).abs() <= 1e-12 {
        return Some(1.0 / (capacity as f64 + 1.0));
    }

    let numerator = 1.0 - rho;
    let denominator = 1.0 - rho.powi(capacity as i32 + 1);
    Some(numerator / denominator)
}

/// Probability an arrival is blocked in M/M/1/K.
pub fn mm1k_blocking_probability(
    arrival_rate: f64,
    service_rate: f64,
    capacity: usize,
) -> Option<f64> {
    let p0 = mm1k_probability_zero(arrival_rate, service_rate, capacity)?;
    let rho = utilization(arrival_rate, service_rate)?;

    if capacity == 0 {
        return Some(1.0);
    }
    if (rho - 1.0).abs() <= 1e-12 {
        return Some(p0);
    }

    Some(p0 * rho.powi(capacity as i32))
}

/// Effective admitted arrival rate in M/M/1/K.
pub fn mm1k_effective_arrival_rate(
    arrival_rate: f64,
    service_rate: f64,
    capacity: usize,
) -> Option<f64> {
    let p_block = mm1k_blocking_probability(arrival_rate, service_rate, capacity)?;
    Some(arrival_rate * (1.0 - p_block))
}

/// Mean number in system for M/M/1/K.
pub fn mm1k_mean_number_in_system(
    arrival_rate: f64,
    service_rate: f64,
    capacity: usize,
) -> Option<f64> {
    let rho = utilization(arrival_rate, service_rate)?;

    if capacity == 0 {
        return Some(0.0);
    }
    if (rho - 1.0).abs() <= 1e-12 {
        return Some(capacity as f64 / 2.0);
    }

    let numerator = rho
        * (1.0 - (capacity as f64 + 1.0) * rho.powi(capacity as i32)
            + capacity as f64 * rho.powi(capacity as i32 + 1));
    let denominator = (1.0 - rho) * (1.0 - rho.powi(capacity as i32 + 1));
    Some(numerator / denominator)
}

/// Mean number waiting in queue for M/M/1/K.
pub fn mm1k_mean_number_in_queue(
    arrival_rate: f64,
    service_rate: f64,
    capacity: usize,
) -> Option<f64> {
    let l = mm1k_mean_number_in_system(arrival_rate, service_rate, capacity)?;
    let utilization_effective = mm1k_effective_arrival_rate(arrival_rate, service_rate, capacity)? / service_rate;
    Some((l - utilization_effective).max(0.0))
}

/// Mean time in system for admitted jobs in M/M/1/K.
pub fn mm1k_mean_time_in_system(
    arrival_rate: f64,
    service_rate: f64,
    capacity: usize,
) -> Option<f64> {
    let l = mm1k_mean_number_in_system(arrival_rate, service_rate, capacity)?;
    let lambda_eff = mm1k_effective_arrival_rate(arrival_rate, service_rate, capacity)?;
    if lambda_eff <= 0.0 {
        return Some(0.0);
    }
    littles_w(l, lambda_eff)
}

/// Mean waiting time in queue for admitted jobs in M/M/1/K.
pub fn mm1k_mean_waiting_time_in_queue(
    arrival_rate: f64,
    service_rate: f64,
    capacity: usize,
) -> Option<f64> {
    let lq = mm1k_mean_number_in_queue(arrival_rate, service_rate, capacity)?;
    let lambda_eff = mm1k_effective_arrival_rate(arrival_rate, service_rate, capacity)?;
    if lambda_eff <= 0.0 {
        return Some(0.0);
    }
    littles_w(lq, lambda_eff)
}

/// Grouped steady-state metrics for M/M/1.
pub fn mm1_metrics(arrival_rate: f64, service_rate: f64) -> Option<QueueMetrics> {
    Some(QueueMetrics {
        mean_number_in_queue: mm1_mean_number_in_queue(arrival_rate, service_rate)?,
        mean_number_in_system: mm1_mean_number_in_system(arrival_rate, service_rate)?,
        mean_waiting_time_in_queue: mm1_mean_waiting_time_in_queue(arrival_rate, service_rate)?,
        mean_time_in_system: mm1_mean_time_in_system(arrival_rate, service_rate)?,
    })
}

/// Grouped steady-state metrics for M/M/c.
pub fn mmc_metrics(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
) -> Option<MultiServerQueueMetrics> {
    let probability_zero = mmc_probability_zero(arrival_rate, service_rate, servers)?;
    let wait_probability = mmc_wait_probability(arrival_rate, service_rate, servers)?;
    let immediate_service_probability = 1.0 - wait_probability;
    let mean_number_in_queue = mmc_mean_number_in_queue(arrival_rate, service_rate, servers)?;
    let mean_waiting_time_in_queue =
        mmc_mean_waiting_time_in_queue(arrival_rate, service_rate, servers)?;
    let mean_time_in_system = mmc_mean_time_in_system(arrival_rate, service_rate, servers)?;
    let mean_number_in_system = mmc_mean_number_in_system(arrival_rate, service_rate, servers)?;

    Some(MultiServerQueueMetrics {
        probability_zero,
        wait_probability,
        immediate_service_probability,
        mean_number_in_queue,
        mean_number_in_system,
        mean_waiting_time_in_queue,
        mean_time_in_system,
    })
}

/// Grouped steady-state metrics for M/M/1/K.
pub fn mm1k_metrics(
    arrival_rate: f64,
    service_rate: f64,
    capacity: usize,
) -> Option<FiniteCapacityQueueMetrics> {
    Some(FiniteCapacityQueueMetrics {
        probability_zero: mm1k_probability_zero(arrival_rate, service_rate, capacity)?,
        blocking_probability: mm1k_blocking_probability(arrival_rate, service_rate, capacity)?,
        effective_arrival_rate: mm1k_effective_arrival_rate(arrival_rate, service_rate, capacity)?,
        mean_number_in_queue: mm1k_mean_number_in_queue(arrival_rate, service_rate, capacity)?,
        mean_number_in_system: mm1k_mean_number_in_system(arrival_rate, service_rate, capacity)?,
        mean_waiting_time_in_queue: mm1k_mean_waiting_time_in_queue(arrival_rate, service_rate, capacity)?,
        mean_time_in_system: mm1k_mean_time_in_system(arrival_rate, service_rate, capacity)?,
    })
}

/// Grouped steady-state metrics for Erlang A (M/M/c+M).
pub fn erlang_a_metrics(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<ErlangAMetrics> {
    Some(ErlangAMetrics {
        probability_zero: erlang_a_probability_zero(arrival_rate, service_rate, servers, abandonment_rate)?,
        delay_probability: erlang_a_delay_probability(arrival_rate, service_rate, servers, abandonment_rate)?,
        service_probability: erlang_a_service_probability(arrival_rate, service_rate, servers, abandonment_rate)?,
        abandonment_probability: erlang_a_abandonment_probability(arrival_rate, service_rate, servers, abandonment_rate)?,
        mean_number_in_queue: erlang_a_mean_number_in_queue(arrival_rate, service_rate, servers, abandonment_rate)?,
        mean_number_in_system: erlang_a_mean_number_in_system(arrival_rate, service_rate, servers, abandonment_rate)?,
        mean_waiting_time_in_queue: erlang_a_mean_waiting_time_in_queue(arrival_rate, service_rate, servers, abandonment_rate)?,
        mean_time_in_system: erlang_a_mean_time_in_system(arrival_rate, service_rate, servers, abandonment_rate)?,
    })
}

/// Probability that an M/M/c+M (Erlang A) system is empty.
pub fn erlang_a_probability_zero(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<f64> {
    let (base_mass, tail_mass, p_c) =
        erlang_a_components(arrival_rate, service_rate, servers, abandonment_rate)?;
    Some(1.0 / (base_mass + p_c * tail_mass))
}

/// Delay probability for M/M/c+M (Erlang A).
pub fn erlang_a_delay_probability(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<f64> {
    let (base_mass, tail_mass, p_c) =
        erlang_a_components(arrival_rate, service_rate, servers, abandonment_rate)?;
    let p0 = 1.0 / (base_mass + p_c * tail_mass);
    Some(p0 * p_c * tail_mass)
}

/// Mean queue length for M/M/c+M (Erlang A).
pub fn erlang_a_mean_number_in_queue(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<f64> {
    let (base_mass, _tail_mass, p_c) =
        erlang_a_components(arrival_rate, service_rate, servers, abandonment_rate)?;
    let tail_mass = erlang_a_tail_mass_sum(arrival_rate, service_rate, servers, abandonment_rate)?;
    let p0 = 1.0 / (base_mass + p_c * tail_mass);
    let queue_tail = erlang_a_queue_tail_sum(arrival_rate, service_rate, servers, abandonment_rate)?;
    Some(p0 * p_c * queue_tail)
}

/// Mean waiting time in queue for M/M/c+M, counting all arrivals until service or abandonment.
pub fn erlang_a_mean_waiting_time_in_queue(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<f64> {
    if arrival_rate == 0.0 {
        return Some(0.0);
    }
    let lq = erlang_a_mean_number_in_queue(arrival_rate, service_rate, servers, abandonment_rate)?;
    littles_w(lq, arrival_rate)
}

/// Probability an arrival is eventually served in M/M/c+M.
pub fn erlang_a_service_probability(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<f64> {
    if arrival_rate == 0.0 {
        return Some(1.0);
    }
    let lq = erlang_a_mean_number_in_queue(arrival_rate, service_rate, servers, abandonment_rate)?;
    let abandon_probability = abandonment_rate * lq / arrival_rate;
    Some((1.0 - abandon_probability).clamp(0.0, 1.0))
}

/// Probability an arrival abandons before service in M/M/c+M.
pub fn erlang_a_abandonment_probability(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<f64> {
    let service_probability = erlang_a_service_probability(arrival_rate, service_rate, servers, abandonment_rate)?;
    Some(1.0 - service_probability)
}

/// Mean number in system for M/M/c+M.
pub fn erlang_a_mean_number_in_system(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<f64> {
    let lq = erlang_a_mean_number_in_queue(arrival_rate, service_rate, servers, abandonment_rate)?;
    let p_service = erlang_a_service_probability(arrival_rate, service_rate, servers, abandonment_rate)?;
    Some(lq + arrival_rate * p_service / service_rate)
}

/// Mean time in system for M/M/c+M, counting all arrivals until service completion or abandonment.
pub fn erlang_a_mean_time_in_system(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<f64> {
    let wq = erlang_a_mean_waiting_time_in_queue(arrival_rate, service_rate, servers, abandonment_rate)?;
    let p_service = erlang_a_service_probability(arrival_rate, service_rate, servers, abandonment_rate)?;
    Some(wq + p_service / service_rate)
}

fn erlang_a_components(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<(f64, f64, f64)> {
    if servers == 0
        || arrival_rate < 0.0
        || service_rate <= 0.0
        || abandonment_rate <= 0.0
        || !arrival_rate.is_finite()
        || !service_rate.is_finite()
        || !abandonment_rate.is_finite()
    {
        return None;
    }

    let a = arrival_rate / service_rate;
    let mut base_mass = 0.0;
    for n in 0..servers {
        base_mass += a.powi(n as i32) / factorial(n);
    }

    let p_c = a.powi(servers as i32) / factorial(servers);
    let tail_mass = erlang_a_tail_mass_sum(arrival_rate, service_rate, servers, abandonment_rate)?;

    Some((base_mass, tail_mass, p_c))
}

fn erlang_a_tail_mass_sum(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<f64> {
    let mut term = 1.0;
    let mut sum = 1.0;

    for j in 1..=ERLANG_A_MAX_TERMS {
        term *= arrival_rate / (servers as f64 * service_rate + j as f64 * abandonment_rate);
        sum += term;
        if term.abs() < ERLANG_A_TERM_TOL {
            return Some(sum);
        }
    }

    None
}

fn erlang_a_queue_tail_sum(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    abandonment_rate: f64,
) -> Option<f64> {
    let mut term = 1.0;
    let mut sum = 0.0;

    for j in 1..=ERLANG_A_MAX_TERMS {
        term *= arrival_rate / (servers as f64 * service_rate + j as f64 * abandonment_rate);
        sum += j as f64 * term;
        if term.abs() < ERLANG_A_TERM_TOL {
            return Some(sum);
        }
    }

    None
}

fn factorial(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    (1..=n).fold(1.0, |acc, value| acc * value as f64)
}

const ERLANG_A_MAX_TERMS: usize = 100_000;
const ERLANG_A_TERM_TOL: f64 = 1e-14;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn little_law_round_trip() {
        let l = littles_l(2.0, 1.5);
        let w = littles_w(l, 2.0).unwrap();
        assert!((l - 3.0).abs() < 1e-12);
        assert!((w - 1.5).abs() < 1e-12);
    }

    #[test]
    fn grouped_mm1_metrics_match_scalar_helpers() {
        let metrics = mm1_metrics(2.0, 3.0).unwrap();
        assert!((metrics.mean_number_in_queue - 4.0 / 3.0).abs() < 1e-12);
        assert!((metrics.mean_number_in_system - 2.0).abs() < 1e-12);
        assert!((metrics.mean_waiting_time_in_queue - 2.0 / 3.0).abs() < 1e-12);
        assert!((metrics.mean_time_in_system - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mm1_formulas_match() {
        let lq = mm1_mean_number_in_queue(2.0, 3.0).unwrap();
        let l = mm1_mean_number_in_system(2.0, 3.0).unwrap();
        let wq = mm1_mean_waiting_time_in_queue(2.0, 3.0).unwrap();
        let w = mm1_mean_time_in_system(2.0, 3.0).unwrap();
        assert!((lq - 4.0 / 3.0).abs() < 1e-12);
        assert!((l - 2.0).abs() < 1e-12);
        assert!((wq - 2.0 / 3.0).abs() < 1e-12);
        assert!((w - 1.0).abs() < 1e-12);
        assert!((lq - littles_l(2.0, wq)).abs() < 1e-12);
        assert!((l - littles_l(2.0, w)).abs() < 1e-12);
    }

    #[test]
    fn unstable_mm1_returns_none() {
        assert_eq!(mm1_mean_number_in_queue(3.0, 3.0), None);
        assert_eq!(mm1_mean_number_in_system(3.0, 3.0), None);
        assert_eq!(mm1_mean_waiting_time_in_queue(3.0, 3.0), None);
        assert_eq!(mm1_mean_time_in_system(3.0, 3.0), None);
    }

    #[test]
    fn erlang_b_known_case() {
        let blocking = erlang_b(10.0, 10).unwrap();
        assert!((blocking - 0.2145823431).abs() < 1e-9);
    }

    #[test]
    fn mmc_probabilities_and_delays_are_consistent() {
        let lambda = 4.0;
        let mu = 3.0;
        let c = 2;

        let p0 = mmc_probability_zero(lambda, mu, c).unwrap();
        let pwait = erlang_c(lambda, mu, c).unwrap();
        let lq = mmc_mean_number_in_queue(lambda, mu, c).unwrap();
        let wq = mmc_mean_waiting_time_in_queue(lambda, mu, c).unwrap();
        let w = mmc_mean_time_in_system(lambda, mu, c).unwrap();
        let l = mmc_mean_number_in_system(lambda, mu, c).unwrap();

        assert!((p0 - 0.2).abs() < 1e-12);
        assert!((pwait - 0.5333333333333333).abs() < 1e-12);
        assert!((lq - 1.0666666666666664).abs() < 1e-12);
        assert!((wq - 0.2666666666666666).abs() < 1e-12);
        assert!((w - 0.6).abs() < 1e-12);
        assert!((l - 2.4).abs() < 1e-12);
        assert!((l - littles_l(lambda, w)).abs() < 1e-12);
    }

    #[test]
    fn unstable_mmc_returns_none() {
        assert_eq!(mmc_probability_zero(6.0, 3.0, 2), None);
        assert_eq!(erlang_c(6.0, 3.0, 2), None);
        assert_eq!(mmc_mean_number_in_queue(6.0, 3.0, 2), None);
        assert_eq!(mmc_wait_exceeds_probability(6.0, 3.0, 2, 1.0), None);
        assert_eq!(mmc_service_level(6.0, 3.0, 2, 1.0), None);
    }

    #[test]
    fn mmc_service_level_helpers_are_consistent() {
        let lambda = 4.0;
        let mu = 3.0;
        let c = 2;
        let t = 1.0;

        let metrics = mmc_metrics(lambda, mu, c).unwrap();
        let pwait = mmc_wait_probability(lambda, mu, c).unwrap();
        let p0wait = mmc_wait_exceeds_probability(lambda, mu, c, 0.0).unwrap();
        let immediate = mmc_immediate_service_probability(lambda, mu, c).unwrap();
        let sla = mmc_service_level(lambda, mu, c, t).unwrap();

        assert!((metrics.wait_probability - pwait).abs() < 1e-12);
        assert!((metrics.immediate_service_probability - immediate).abs() < 1e-12);
        assert!((pwait - p0wait).abs() < 1e-12);
        assert!((immediate - (1.0 - pwait)).abs() < 1e-12);
        assert!(sla > immediate);
        assert!(sla < 1.0);
    }

    #[test]
    fn mmc_system_time_distribution_is_well_behaved() {
        let lambda = 4.0;
        let mu = 3.0;
        let c = 2;

        let cdf0 = mmc_system_time_cdf(lambda, mu, c, 0.0).unwrap();
        let cdf1 = mmc_system_time_cdf(lambda, mu, c, 1.0).unwrap();
        let cdf2 = mmc_system_time_cdf(lambda, mu, c, 5.0).unwrap();
        let tail1 = mmc_system_time_exceeds_probability(lambda, mu, c, 1.0).unwrap();

        assert!((cdf0 - 0.0).abs() < 1e-12);
        assert!(cdf1 > cdf0);
        assert!(cdf2 > cdf1);
        assert!(cdf2 < 1.0);
        assert!((tail1 - (1.0 - cdf1)).abs() < 1e-12);
    }

    #[test]
    fn mm1k_formulas_are_consistent() {
        let lambda = 2.0;
        let mu = 3.0;
        let k = 4;

        let metrics = mm1k_metrics(lambda, mu, k).unwrap();
        let p0 = mm1k_probability_zero(lambda, mu, k).unwrap();
        let p_block = mm1k_blocking_probability(lambda, mu, k).unwrap();
        let lambda_eff = mm1k_effective_arrival_rate(lambda, mu, k).unwrap();
        let l = mm1k_mean_number_in_system(lambda, mu, k).unwrap();
        let lq = mm1k_mean_number_in_queue(lambda, mu, k).unwrap();
        let w = mm1k_mean_time_in_system(lambda, mu, k).unwrap();
        let wq = mm1k_mean_waiting_time_in_queue(lambda, mu, k).unwrap();

        assert!((metrics.probability_zero - p0).abs() < 1e-12);
        assert!((metrics.blocking_probability - p_block).abs() < 1e-12);
        assert!((metrics.effective_arrival_rate - lambda_eff).abs() < 1e-12);
        assert!((metrics.mean_number_in_system - l).abs() < 1e-12);
        assert!((metrics.mean_number_in_queue - lq).abs() < 1e-12);
        assert!((metrics.mean_time_in_system - w).abs() < 1e-12);
        assert!((metrics.mean_waiting_time_in_queue - wq).abs() < 1e-12);
        assert!((0.0..=1.0).contains(&p0));
        assert!((0.0..=1.0).contains(&p_block));
        assert!(lambda_eff <= lambda + 1e-12);
        assert!(l >= lq);
        assert!((l - littles_l(lambda_eff, w)).abs() < 1e-9);
        assert!((lq - littles_l(lambda_eff, wq)).abs() < 1e-9);
        assert!((w - (wq + 1.0 / mu)).abs() < 1e-9);
    }

    #[test]
    fn mm1k_zero_capacity_is_trivial_loss_system() {
        let p0 = mm1k_probability_zero(2.0, 3.0, 0).unwrap();
        let p_block = mm1k_blocking_probability(2.0, 3.0, 0).unwrap();
        let lambda_eff = mm1k_effective_arrival_rate(2.0, 3.0, 0).unwrap();
        let l = mm1k_mean_number_in_system(2.0, 3.0, 0).unwrap();
        let w = mm1k_mean_time_in_system(2.0, 3.0, 0).unwrap();

        assert!((p0 - 1.0).abs() < 1e-12);
        assert!((p_block - 1.0).abs() < 1e-12);
        assert!((lambda_eff - 0.0).abs() < 1e-12);
        assert!((l - 0.0).abs() < 1e-12);
        assert!((w - 0.0).abs() < 1e-12);
    }

    #[test]
    fn erlang_a_zero_arrival_is_trivial() {
        let p0 = erlang_a_probability_zero(0.0, 3.0, 2, 1.0).unwrap();
        let pwait = erlang_a_delay_probability(0.0, 3.0, 2, 1.0).unwrap();
        let lq = erlang_a_mean_number_in_queue(0.0, 3.0, 2, 1.0).unwrap();
        let wq = erlang_a_mean_waiting_time_in_queue(0.0, 3.0, 2, 1.0).unwrap();
        let p_service = erlang_a_service_probability(0.0, 3.0, 2, 1.0).unwrap();
        let p_abandon = erlang_a_abandonment_probability(0.0, 3.0, 2, 1.0).unwrap();

        assert!((p0 - 1.0).abs() < 1e-12);
        assert!((pwait - 0.0).abs() < 1e-12);
        assert!((lq - 0.0).abs() < 1e-12);
        assert!((wq - 0.0).abs() < 1e-12);
        assert!((p_service - 1.0).abs() < 1e-12);
        assert!((p_abandon - 0.0).abs() < 1e-12);
    }

    #[test]
    fn erlang_a_metrics_are_probabilities_and_consistent() {
        let lambda = 4.0;
        let mu = 3.0;
        let c = 2;
        let theta = 1.0;

        let metrics = erlang_a_metrics(lambda, mu, c, theta).unwrap();
        let p0 = erlang_a_probability_zero(lambda, mu, c, theta).unwrap();
        let pwait = erlang_a_delay_probability(lambda, mu, c, theta).unwrap();
        let lq = erlang_a_mean_number_in_queue(lambda, mu, c, theta).unwrap();
        let wq = erlang_a_mean_waiting_time_in_queue(lambda, mu, c, theta).unwrap();
        let p_service = erlang_a_service_probability(lambda, mu, c, theta).unwrap();
        let p_abandon = erlang_a_abandonment_probability(lambda, mu, c, theta).unwrap();
        let l = erlang_a_mean_number_in_system(lambda, mu, c, theta).unwrap();
        let w = erlang_a_mean_time_in_system(lambda, mu, c, theta).unwrap();

        assert!((metrics.probability_zero - p0).abs() < 1e-12);
        assert!((metrics.delay_probability - pwait).abs() < 1e-12);
        assert!((metrics.service_probability - p_service).abs() < 1e-12);
        assert!((metrics.abandonment_probability - p_abandon).abs() < 1e-12);
        assert!((metrics.mean_number_in_queue - lq).abs() < 1e-12);
        assert!((metrics.mean_waiting_time_in_queue - wq).abs() < 1e-12);
        assert!((metrics.mean_number_in_system - l).abs() < 1e-12);
        assert!((metrics.mean_time_in_system - w).abs() < 1e-12);
        assert!((0.0..=1.0).contains(&p0));
        assert!((0.0..=1.0).contains(&pwait));
        assert!((0.0..=1.0).contains(&p_service));
        assert!((0.0..=1.0).contains(&p_abandon));
        assert!((p_service + p_abandon - 1.0).abs() < 1e-9);
        assert!((lq - littles_l(lambda, wq)).abs() < 1e-9);
        assert!((l - littles_l(lambda, w)).abs() < 1e-9);
        assert!((l - (lq + lambda * p_service / mu)).abs() < 1e-9);
    }

    #[test]
    fn erlang_a_more_abandonment_reduces_queue_length() {
        let low_theta = erlang_a_mean_number_in_queue(8.0, 3.0, 3, 0.5).unwrap();
        let high_theta = erlang_a_mean_number_in_queue(8.0, 3.0, 3, 2.0).unwrap();
        assert!(high_theta < low_theta);

        let low_abandon = erlang_a_abandonment_probability(8.0, 3.0, 3, 0.5).unwrap();
        let high_abandon = erlang_a_abandonment_probability(8.0, 3.0, 3, 2.0).unwrap();
        assert!(high_abandon > low_abandon);
    }
}
