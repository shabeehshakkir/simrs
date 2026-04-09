//! Quick sanity checks for closed-form queueing helpers.
//!
//! Usage:
//!   cargo run --example queueing_sanity

use simrs::math::queueing::{
    erlang_a_metrics, mm1_metrics, mm1k_metrics, mmc_metrics, mmc_service_level,
};

fn main() {
    println!("== simrs queueing sanity checks ==");

    let mm1 = mm1_metrics(2.0, 3.0).expect("valid M/M/1 parameters");
    println!("M/M/1 (λ=2, μ=3): {:?}", mm1);
    assert!((mm1.mean_number_in_queue - 4.0 / 3.0).abs() < 1e-12);
    assert!((mm1.mean_number_in_system - 2.0).abs() < 1e-12);
    assert!((mm1.mean_waiting_time_in_queue - 2.0 / 3.0).abs() < 1e-12);
    assert!((mm1.mean_time_in_system - 1.0).abs() < 1e-12);

    let mmc = mmc_metrics(4.0, 3.0, 2).expect("valid M/M/c parameters");
    let sla_1 = mmc_service_level(4.0, 3.0, 2, 1.0).expect("valid SLA parameters");
    println!("M/M/2 (λ=4, μ=3): {:?}", mmc);
    println!("M/M/2 service level P(Wq <= 1.0): {:.6}", sla_1);
    assert!((mmc.wait_probability - 0.5333333333333333).abs() < 1e-12);
    assert!(sla_1 > mmc.immediate_service_probability);
    assert!(sla_1 < 1.0);

    let mm1k = mm1k_metrics(2.0, 3.0, 4).expect("valid M/M/1/K parameters");
    println!("M/M/1/4 (λ=2, μ=3): {:?}", mm1k);
    assert!(mm1k.blocking_probability > 0.0);
    assert!(mm1k.blocking_probability < 1.0);
    assert!(mm1k.effective_arrival_rate < 2.0);
    assert!(mm1k.mean_number_in_system >= mm1k.mean_number_in_queue);

    let erlang_a = erlang_a_metrics(4.0, 3.0, 2, 1.0).expect("valid Erlang A parameters");
    println!("Erlang A M/M/2+M (λ=4, μ=3, θ=1): {:?}", erlang_a);
    assert!((erlang_a.service_probability + erlang_a.abandonment_probability - 1.0).abs() < 1e-9);
    assert!(erlang_a.mean_number_in_system >= erlang_a.mean_number_in_queue);

    println!("All queueing sanity checks passed.");
}
