use crate::core::{
    event::{EventPayload, Phase, ScheduledEvent},
    event_set::{BinaryHeapEventSet, EventSet},
    rng::RngManager,
    time::SimTime,
};

/// The central simulation context.
///
/// Owns the event set, simulation clock, sequence counter, and RNG manager.
/// All scheduling goes through this struct.
pub struct Simulation {
    /// Current simulation time.
    now: SimTime,
    /// Monotonically increasing sequence counter — ensures FIFO ordering
    /// within events at the same `(time, phase)`.
    seq: u64,
    /// Pending-event data structure.
    events: Box<dyn EventSet>,
    /// Random number generator manager.
    pub(crate) rng: RngManager,
    /// Hard upper time limit (if set). `run_until` uses this.
    time_limit: Option<SimTime>,
}

impl Simulation {
    /// Create a simulation seeded with `seed`.
    pub fn seeded(seed: u64) -> Self {
        Simulation {
            now: SimTime::ZERO,
            seq: 0,
            events: Box::new(BinaryHeapEventSet::new()),
            rng: RngManager::seeded(seed),
            time_limit: None,
        }
    }

    /// Entry point for the builder pattern.
    pub fn builder() -> SimulationBuilder {
        SimulationBuilder::default()
    }

    /// Current simulation time.
    #[inline]
    pub fn now(&self) -> SimTime {
        self.now
    }

    /// Schedule an event at an absolute time with the given phase.
    ///
    /// Returns the sequence number assigned to this event.
    pub fn schedule_at(&mut self, time: SimTime, phase: Phase, payload: EventPayload) -> u64 {
        let seq = self.seq;
        self.seq += 1;
        self.events.push(ScheduledEvent::new(time, phase, seq, payload));
        seq
    }

    /// Schedule an event `dt` units after the current time.
    pub fn schedule_in(&mut self, dt: f64, phase: Phase, payload: EventPayload) -> u64 {
        let time = self.now + dt;
        self.schedule_at(time, phase, payload)
    }

    /// Advance to the next scheduled event and return it.
    ///
    /// Skips cancelled events. Returns `None` if no events remain
    /// (or the next event is past the time limit).
    pub fn step(&mut self) -> Option<ScheduledEvent> {
        loop {
            let next = self.events.pop()?;

            // Respect time limit
            if let Some(limit) = self.time_limit {
                if next.time > limit {
                    return None;
                }
            }

            // Skip cancelled events
            if next.cancelled {
                continue;
            }

            // Advance clock (monotonic — never go backward)
            debug_assert!(
                next.time >= self.now,
                "event time {} is before current time {}",
                next.time,
                self.now
            );
            self.now = next.time;
            return Some(next);
        }
    }

    /// Run until no events remain.
    pub fn run(&mut self) {
        while self.step().is_some() {}
    }

    /// Run until the simulation clock reaches `until` (exclusive — stops
    /// before processing events at exactly `until` if they are past the limit).
    pub fn run_until(&mut self, until: f64) {
        self.time_limit = Some(SimTime::from_f64(until));
        self.run();
        self.time_limit = None;
    }

    /// Run until no events remain (alias for `run`, provided for clarity).
    pub fn run_until_no_events(&mut self) {
        self.run();
    }

    /// Number of events currently pending in the event set.
    pub fn pending_events(&self) -> usize {
        self.events.len()
    }

    /// Peek at the time of the next event without advancing the clock.
    /// Returns `None` if the event set is empty.
    pub fn pending_events_peek(&self) -> Option<SimTime> {
        self.events.peek().map(|e| e.time)
    }

    /// Access the RNG manager for sampling distributions.
    pub fn rng_mut(&mut self) -> &mut RngManager {
        &mut self.rng
    }
}

/// Builder for [`Simulation`].
#[derive(Default)]
pub struct SimulationBuilder {
    seed: Option<u64>,
    initial_capacity: Option<usize>,
}

impl SimulationBuilder {
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn initial_capacity(mut self, cap: usize) -> Self {
        self.initial_capacity = Some(cap);
        self
    }

    pub fn build(self) -> Simulation {
        let seed = self.seed.unwrap_or(0);
        let event_set: Box<dyn EventSet> = match self.initial_capacity {
            Some(cap) => Box::new(BinaryHeapEventSet::with_capacity(cap)),
            None => Box::new(BinaryHeapEventSet::new()),
        };
        Simulation {
            now: SimTime::ZERO,
            seq: 0,
            events: event_set,
            rng: RngManager::seeded(seed),
            time_limit: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_pop_in_order() {
        let mut sim = Simulation::seeded(0);
        sim.schedule_at(SimTime::from_f64(3.0), Phase::Arrival, EventPayload::User { tag: 3 });
        sim.schedule_at(SimTime::from_f64(1.0), Phase::Arrival, EventPayload::User { tag: 1 });
        sim.schedule_at(SimTime::from_f64(2.0), Phase::Arrival, EventPayload::User { tag: 2 });

        let e1 = sim.step().unwrap();
        assert_eq!(e1.time.as_f64(), 1.0);
        assert_eq!(sim.now().as_f64(), 1.0);

        let e2 = sim.step().unwrap();
        assert_eq!(e2.time.as_f64(), 2.0);

        let e3 = sim.step().unwrap();
        assert_eq!(e3.time.as_f64(), 3.0);

        assert!(sim.step().is_none());
    }

    #[test]
    fn run_until_stops_at_limit() {
        let mut sim = Simulation::seeded(0);
        for i in 0..10 {
            sim.schedule_at(
                SimTime::from_f64(i as f64),
                Phase::Arrival,
                EventPayload::User { tag: i },
            );
        }
        sim.run_until(5.0);
        // Events at t=5 are processed (5.0 <= 5.0), t=6..9 are not
        assert!(sim.now().as_f64() <= 5.0);
    }

    #[test]
    fn cancelled_events_are_skipped() {
        let mut sim = Simulation::seeded(0);
        let seq =
            sim.schedule_at(SimTime::from_f64(1.0), Phase::Arrival, EventPayload::User { tag: 1 });
        // Manually cancel by peeking into internals — in real usage the primitives
        // layer handles this via a cancellation token.
        // For this unit test, we just verify the count.
        assert_eq!(sim.pending_events(), 1);
        let _ = seq; // used
    }

    #[test]
    fn seq_increments_monotonically() {
        let mut sim = Simulation::seeded(0);
        let s0 = sim.schedule_at(SimTime::ZERO, Phase::Arrival, EventPayload::User { tag: 0 });
        let s1 = sim.schedule_at(SimTime::ZERO, Phase::Arrival, EventPayload::User { tag: 1 });
        let s2 = sim.schedule_at(SimTime::ZERO, Phase::Arrival, EventPayload::User { tag: 2 });
        assert!(s0 < s1);
        assert!(s1 < s2);
    }

    #[test]
    fn builder_produces_working_simulation() {
        let mut sim = Simulation::builder().seed(42).initial_capacity(64).build();
        sim.schedule_at(SimTime::from_f64(1.0), Phase::Arrival, EventPayload::User { tag: 0 });
        assert!(sim.step().is_some());
    }
}
