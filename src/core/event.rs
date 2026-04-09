use crate::core::time::SimTime;
use std::cmp::Ordering;

/// Execution phase for same-time event ordering.
///
/// Lower numeric values have higher priority (fire first).
/// This prevents accidental semantic ambiguity when multiple events
/// are scheduled at the same simulation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Phase {
    /// Process interrupts — highest priority, fire first.
    Interrupt = 0,
    /// Resource release events — typically fire before Resume to allow
    /// waiting processes to be granted the resource in the same time step.
    Release = 1,
    /// Process resume / wake events.
    Resume = 2,
    /// Service completion events.
    ServiceCompletion = 3,
    /// Entity/customer arrival events.
    Arrival = 4,
    /// Monitor/statistics collection events.
    Monitor = 5,
    /// User-defined events — lowest priority, fire last at any given time.
    UserDefined = 6,
}

impl Default for Phase {
    fn default() -> Self {
        Phase::UserDefined
    }
}

/// Opaque event payload stored on the event queue.
///
/// The kernel treats this as opaque data; the primitives layer interprets it.
#[derive(Debug, Clone)]
pub enum EventPayload {
    /// A process wake event carrying the process ID.
    ProcessWake { process_id: u64 },
    /// A resource grant event.
    ResourceGrant { resource_id: u64, waiter_id: u64 },
    /// A generic user event with an optional u64 tag.
    User { tag: u64 },
    /// An end-of-simulation sentinel.
    EndOfSimulation,
}

/// An event scheduled to fire at a specific simulation time.
///
/// Events are totally ordered by `(time, phase, seq)` to guarantee
/// deterministic same-time event ordering under fixed seeds.
#[derive(Debug, Clone)]
pub struct ScheduledEvent {
    pub time: SimTime,
    pub phase: Phase,
    /// Monotonically increasing sequence number assigned by the `Simulation`.
    /// Guarantees FIFO ordering for events at the same `(time, phase)`.
    pub seq: u64,
    pub payload: EventPayload,
    /// If `true`, this event should be skipped when popped from the event set.
    pub cancelled: bool,
}

impl ScheduledEvent {
    pub fn new(time: SimTime, phase: Phase, seq: u64, payload: EventPayload) -> Self {
        ScheduledEvent { time, phase, seq, payload, cancelled: false }
    }
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.phase == other.phase && self.seq == other.seq
    }
}

impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: time ascending
        self.time
            .cmp(&other.time)
            // Secondary: phase ascending (lower value = higher priority)
            .then(self.phase.cmp(&other.phase))
            // Tertiary: seq ascending (FIFO within same time+phase)
            .then(self.seq.cmp(&other.seq))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(time: f64, phase: Phase, seq: u64) -> ScheduledEvent {
        ScheduledEvent::new(SimTime::from_f64(time), phase, seq, EventPayload::User { tag: 0 })
    }

    #[test]
    fn time_ordering() {
        let a = make(1.0, Phase::Arrival, 0);
        let b = make(2.0, Phase::Arrival, 1);
        assert!(a < b);
    }

    #[test]
    fn phase_ordering_at_same_time() {
        let interrupt = make(5.0, Phase::Interrupt, 0);
        let arrival = make(5.0, Phase::Arrival, 1);
        // Interrupt fires before Arrival at the same time
        assert!(interrupt < arrival);
    }

    #[test]
    fn seq_ordering_at_same_time_and_phase() {
        let first = make(3.0, Phase::Arrival, 0);
        let second = make(3.0, Phase::Arrival, 1);
        // Earlier seq fires first — FIFO
        assert!(first < second);
    }

    #[test]
    fn phase_enum_ordering() {
        assert!(Phase::Interrupt < Phase::Release);
        assert!(Phase::Release < Phase::Resume);
        assert!(Phase::Resume < Phase::ServiceCompletion);
        assert!(Phase::ServiceCompletion < Phase::Arrival);
        assert!(Phase::Arrival < Phase::Monitor);
        assert!(Phase::Monitor < Phase::UserDefined);
    }
}
