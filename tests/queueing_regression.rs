use simrs::math::queueing::{
    erlang_a_metrics, mm1_metrics, mm1k_metrics, mmc_metrics, mmc_service_level,
    mmc_system_time_cdf,
};

#[test]
fn mm1_regression_values_match_reference_case() {
    let metrics = mm1_metrics(2.0, 3.0).unwrap();
    assert!((metrics.mean_number_in_queue - 4.0 / 3.0).abs() < 1e-12);
    assert!((metrics.mean_number_in_system - 2.0).abs() < 1e-12);
    assert!((metrics.mean_waiting_time_in_queue - 2.0 / 3.0).abs() < 1e-12);
    assert!((metrics.mean_time_in_system - 1.0).abs() < 1e-12);
}

#[test]
fn mmc_regression_values_match_reference_case() {
    let metrics = mmc_metrics(4.0, 3.0, 2).unwrap();
    assert!((metrics.probability_zero - 0.2).abs() < 1e-12);
    assert!((metrics.wait_probability - 0.5333333333333333).abs() < 1e-12);
    assert!((metrics.immediate_service_probability - 0.4666666666666667).abs() < 1e-12);
    assert!((metrics.mean_number_in_queue - 1.0666666666666664).abs() < 1e-12);
    assert!((metrics.mean_number_in_system - 2.4).abs() < 1e-12);
    assert!((metrics.mean_waiting_time_in_queue - 0.2666666666666666).abs() < 1e-12);
    assert!((metrics.mean_time_in_system - 0.6).abs() < 1e-12);
}

#[test]
fn mmc_sla_regression_case_is_stable() {
    let sla = mmc_service_level(4.0, 3.0, 2, 1.0).unwrap();
    let system_cdf = mmc_system_time_cdf(4.0, 3.0, 2, 1.0).unwrap();
    assert!((sla - 0.927821).abs() < 1e-6);
    assert!(system_cdf < sla);
    assert!(system_cdf > 0.0);
}

#[test]
fn mm1k_regression_case_is_stable() {
    let metrics = mm1k_metrics(2.0, 3.0, 4).unwrap();
    assert!((metrics.probability_zero - 0.38388625592417064).abs() < 1e-12);
    assert!((metrics.blocking_probability - 0.07582938388625593).abs() < 1e-12);
    assert!((metrics.effective_arrival_rate - 1.8483412322274881).abs() < 1e-12);
    assert!((metrics.mean_number_in_queue - 0.6255924170616111).abs() < 1e-12);
    assert!((metrics.mean_number_in_system - 1.2417061611374405).abs() < 1e-12);
}

#[test]
fn erlang_a_regression_case_is_stable() {
    let metrics = erlang_a_metrics(4.0, 3.0, 2, 1.0).unwrap();
    assert!((metrics.probability_zero - 0.2400166508568344).abs() < 1e-12);
    assert!((metrics.delay_probability - 0.43996114800071967).abs() < 1e-12);
    assert!((metrics.service_probability - 0.8999583728579152).abs() < 1e-12);
    assert!((metrics.abandonment_probability - 0.10004162714208475).abs() < 1e-12);
    assert!((metrics.mean_number_in_queue - 0.400166508568339).abs() < 1e-12);
    assert!((metrics.mean_number_in_system - 1.600111005712226).abs() < 1e-12);
}
