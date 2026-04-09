/// Online statistics accumulators for simulation monitoring.
///
/// Both monitor types accumulate observations incrementally without
/// storing the full trace (unless explicitly requested), keeping memory
/// stable for long runs.

// ─── TallyMonitor ─────────────────────────────────────────────────────────────

/// Records independent observations and computes summary statistics online
/// using Welford's single-pass algorithm.
///
/// Use for: waiting times, sojourn times, service durations, batch sizes.
#[derive(Debug, Clone)]
pub struct TallyMonitor {
    name: String,
    count: u64,
    mean: f64,
    m2: f64, // sum of squared deviations from the mean (Welford)
    min: f64,
    max: f64,
    /// Optional trace of all recorded values (disabled by default).
    trace: Option<Vec<f64>>,
    /// Observations recorded before warmup (discarded when reset).
    active: bool,
}

impl TallyMonitor {
    pub fn new(name: impl Into<String>) -> Self {
        TallyMonitor {
            name: name.into(),
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            trace: None,
            active: true,
        }
    }

    /// Enable full trace capture (trades memory for trace access).
    pub fn with_trace(mut self) -> Self {
        self.trace = Some(Vec::new());
        self
    }

    /// Record a new observation.
    pub fn record(&mut self, value: f64) {
        if !self.active {
            return;
        }

        self.count += 1;

        // Welford online mean + M2 update
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        if let Some(trace) = &mut self.trace {
            trace.push(value);
        }
    }

    /// Reset accumulated statistics (used for warm-up deletion).
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        if let Some(trace) = &mut self.trace {
            trace.clear();
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 { None } else { Some(self.mean) }
    }

    /// Unbiased sample variance (denominator n-1).
    pub fn variance(&self) -> Option<f64> {
        if self.count < 2 {
            None
        } else {
            Some(self.m2 / (self.count - 1) as f64)
        }
    }

    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    pub fn std_error(&self) -> Option<f64> {
        self.std_dev().map(|sd| sd / (self.count as f64).sqrt())
    }

    pub fn min(&self) -> Option<f64> {
        if self.count == 0 { None } else { Some(self.min) }
    }

    pub fn max(&self) -> Option<f64> {
        if self.count == 0 { None } else { Some(self.max) }
    }

    pub fn trace(&self) -> Option<&[f64]> {
        self.trace.as_deref()
    }

    /// Return a snapshot summary of this monitor.
    pub fn summary(&self) -> MonitorSummary {
        MonitorSummary {
            name: self.name.clone(),
            kind: MonitorKind::Tally,
            count: self.count,
            mean: self.mean().unwrap_or(f64::NAN),
            std_dev: self.std_dev().unwrap_or(f64::NAN),
            min: self.min().unwrap_or(f64::NAN),
            max: self.max().unwrap_or(f64::NAN),
        }
    }
}

// ─── CounterMonitor ───────────────────────────────────────────────────────────

/// Simple named counter for discrete events like completions, stockouts, or interruptions.
#[derive(Debug, Clone)]
pub struct CounterMonitor {
    name: String,
    count: u64,
}

impl CounterMonitor {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), count: 0 }
    }

    pub fn increment(&mut self) {
        self.count += 1;
    }

    pub fn add(&mut self, delta: u64) {
        self.count += delta;
    }

    pub fn reset(&mut self) {
        self.count = 0;
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

// ─── TimeWeightedMonitor ──────────────────────────────────────────────────────

/// Records changes in a state variable and computes the time-weighted average.
///
/// Use for: queue length, server utilization, inventory level.
///
/// Call `set(now, new_value)` whenever the state changes. The monitor
/// accumulates the area under the piecewise-constant curve.
#[derive(Debug, Clone)]
pub struct TimeWeightedMonitor {
    name: String,
    /// Time of the last `set()` call.
    last_time: f64,
    /// Value in force since `last_time`.
    last_value: f64,
    /// Accumulated area (sum of value × Δt).
    area: f64,
    /// Total observed time span.
    total_time: f64,
    /// Current count of transitions.
    transitions: u64,
    min: f64,
    max: f64,
    /// Optional time-series trace: (time, value) pairs.
    trace: Option<Vec<(f64, f64)>>,
    active: bool,
    /// Time at which this monitor started (after reset/warmup).
    start_time: f64,
}

impl TimeWeightedMonitor {
    pub fn new(name: impl Into<String>) -> Self {
        TimeWeightedMonitor {
            name: name.into(),
            last_time: 0.0,
            last_value: 0.0,
            area: 0.0,
            total_time: 0.0,
            transitions: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            trace: None,
            active: true,
            start_time: 0.0,
        }
    }

    pub fn with_trace(mut self) -> Self {
        self.trace = Some(Vec::new());
        self
    }

    /// Update the monitored state variable at the given simulation time.
    ///
    /// Must be called in non-decreasing time order.
    pub fn set(&mut self, now: f64, value: f64) {
        if !self.active {
            return;
        }

        let dt = now - self.last_time;
        if dt > 0.0 {
            self.area += dt * self.last_value;
            self.total_time += dt;
        }

        self.last_time = now;
        self.last_value = value;
        self.transitions += 1;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        if let Some(trace) = &mut self.trace {
            trace.push((now, value));
        }
    }

    /// Flush pending area up to `now` without changing the state value.
    ///
    /// Call this at the end of a run to include the last interval.
    pub fn flush(&mut self, now: f64) {
        self.set(now, self.last_value);
    }

    /// Reset for warm-up deletion.
    pub fn reset(&mut self, now: f64) {
        self.area = 0.0;
        self.total_time = 0.0;
        self.transitions = 0;
        self.min = self.last_value; // current value carries over
        self.max = self.last_value;
        self.last_time = now;
        self.start_time = now;
        if let Some(trace) = &mut self.trace {
            trace.clear();
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn mean(&self) -> Option<f64> {
        if self.total_time <= 0.0 { None } else { Some(self.area / self.total_time) }
    }

    pub fn min(&self) -> Option<f64> {
        if self.transitions == 0 { None } else { Some(self.min) }
    }

    pub fn max(&self) -> Option<f64> {
        if self.transitions == 0 { None } else { Some(self.max) }
    }

    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    pub fn trace(&self) -> Option<&[(f64, f64)]> {
        self.trace.as_deref()
    }

    pub fn summary(&self) -> MonitorSummary {
        MonitorSummary {
            name: self.name.clone(),
            kind: MonitorKind::TimeWeighted,
            count: self.transitions,
            mean: self.mean().unwrap_or(f64::NAN),
            std_dev: f64::NAN, // not applicable
            min: self.min().unwrap_or(f64::NAN),
            max: self.max().unwrap_or(f64::NAN),
        }
    }
}

// ─── Warehouse / stage KPI structs ───────────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq)]
pub struct WarehouseStageMetrics {
    pub mean_wait: f64,
    pub observations: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct WarehouseKpis {
    pub completed_orders: u64,
    pub mean_cycle_time: f64,
    pub pick_stage: WarehouseStageMetrics,
    pub pack_stage: WarehouseStageMetrics,
    pub inventory_stage: WarehouseStageMetrics,
}

impl WarehouseKpis {
    pub fn from_monitors(
        completed_orders: &CounterMonitor,
        cycle_time: &TallyMonitor,
        pick_wait: &TallyMonitor,
        pack_wait: &TallyMonitor,
        inventory_wait: &TallyMonitor,
    ) -> Self {
        Self {
            completed_orders: completed_orders.count(),
            mean_cycle_time: cycle_time.mean().unwrap_or(0.0),
            pick_stage: WarehouseStageMetrics {
                mean_wait: pick_wait.mean().unwrap_or(0.0),
                observations: pick_wait.count(),
            },
            pack_stage: WarehouseStageMetrics {
                mean_wait: pack_wait.mean().unwrap_or(0.0),
                observations: pack_wait.count(),
            },
            inventory_stage: WarehouseStageMetrics {
                mean_wait: inventory_wait.mean().unwrap_or(0.0),
                observations: inventory_wait.count(),
            },
        }
    }
}

// ─── MonitorSummary ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum MonitorKind {
    Tally,
    TimeWeighted,
}

#[derive(Debug, Clone)]
pub struct MonitorSummary {
    pub name: String,
    pub kind: MonitorKind,
    pub count: u64,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_welford_correctness() {
        let mut m = TallyMonitor::new("wait");
        for &v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            m.record(v);
        }
        assert_eq!(m.count(), 8);
        assert!((m.mean().unwrap() - 5.0).abs() < 1e-10);
        let expected_var = 32.0 / 7.0;
        assert!((m.variance().unwrap() - expected_var).abs() < 1e-10);
        assert_eq!(m.min().unwrap(), 2.0);
        assert_eq!(m.max().unwrap(), 9.0);
    }

    #[test]
    fn tally_reset_clears_state() {
        let mut m = TallyMonitor::new("x");
        m.record(1.0);
        m.record(2.0);
        m.reset();
        assert_eq!(m.count(), 0);
        assert!(m.mean().is_none());
    }

    #[test]
    fn counter_monitor_counts_events() {
        let mut c = CounterMonitor::new("completed");
        c.increment();
        c.add(4);
        assert_eq!(c.count(), 5);
        c.reset();
        assert_eq!(c.count(), 0);
    }

    #[test]
    fn time_weighted_mean_flat_signal() {
        let mut m = TimeWeightedMonitor::new("queue_len");
        m.set(0.0, 3.0);
        m.flush(10.0);
        assert!((m.mean().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn time_weighted_mean_step_signal() {
        let mut m = TimeWeightedMonitor::new("inv");
        m.set(0.0, 0.0);
        m.set(4.0, 10.0);
        m.flush(10.0);
        assert!((m.mean().unwrap() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn tally_std_error() {
        let mut m = TallyMonitor::new("x");
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            m.record(v);
        }
        let se = m.std_error().unwrap();
        let expected = m.std_dev().unwrap() / 5.0_f64.sqrt();
        assert!((se - expected).abs() < 1e-12);
    }

    #[test]
    fn warehouse_kpis_build_from_monitors() {
        let mut completed = CounterMonitor::new("completed");
        completed.add(3);
        let mut cycle = TallyMonitor::new("cycle");
        cycle.record(10.0);
        cycle.record(12.0);
        cycle.record(14.0);
        let mut pick = TallyMonitor::new("pick");
        pick.record(1.0);
        pick.record(2.0);
        let mut pack = TallyMonitor::new("pack");
        pack.record(0.5);
        let mut inventory = TallyMonitor::new("inventory");
        inventory.record(0.25);

        let kpis = WarehouseKpis::from_monitors(&completed, &cycle, &pick, &pack, &inventory);
        assert_eq!(kpis.completed_orders, 3);
        assert!((kpis.mean_cycle_time - 12.0).abs() < 1e-12);
        assert_eq!(kpis.pick_stage.observations, 2);
        assert_eq!(kpis.pack_stage.observations, 1);
    }
}
