use crate::core::event::ScheduledEvent;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Abstraction over the pending-event data structure.
///
/// Keeping this as a trait allows swapping implementations (binary heap,
/// calendar queue, bucket queue) via feature flags or benchmarks without
/// changing the `Simulation` or any caller code.
pub trait EventSet {
    /// Insert an event into the set.
    fn push(&mut self, event: ScheduledEvent);

    /// Remove and return the event with the smallest `(time, phase, seq)` key,
    /// or `None` if the set is empty.
    fn pop(&mut self) -> Option<ScheduledEvent>;

    /// Peek at the event with the smallest key without removing it.
    fn peek(&self) -> Option<&ScheduledEvent>;

    /// Number of events currently in the set (including cancelled ones).
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Binary-heap–backed event set.
///
/// Uses `std::collections::BinaryHeap` (max-heap) with `Reverse` to obtain
/// min-heap semantics. Events are popped in ascending `(time, phase, seq)` order.
///
/// This is the default implementation. Alternative structures (calendar queue,
/// bucket queue) may be added and selected via feature flag once benchmark
/// evidence justifies them.
pub struct BinaryHeapEventSet {
    heap: BinaryHeap<Reverse<ScheduledEvent>>,
}

impl BinaryHeapEventSet {
    pub fn new() -> Self {
        BinaryHeapEventSet { heap: BinaryHeap::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        BinaryHeapEventSet { heap: BinaryHeap::with_capacity(cap) }
    }
}

impl Default for BinaryHeapEventSet {
    fn default() -> Self {
        Self::new()
    }
}

impl EventSet for BinaryHeapEventSet {
    fn push(&mut self, event: ScheduledEvent) {
        self.heap.push(Reverse(event));
    }

    fn pop(&mut self) -> Option<ScheduledEvent> {
        self.heap.pop().map(|Reverse(e)| e)
    }

    fn peek(&self) -> Option<&ScheduledEvent> {
        self.heap.peek().map(|Reverse(e)| e)
    }

    fn len(&self) -> usize {
        self.heap.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::event::{EventPayload, Phase};
    use crate::core::time::SimTime;

    fn event(time: f64, phase: Phase, seq: u64) -> ScheduledEvent {
        ScheduledEvent::new(
            SimTime::from_f64(time),
            phase,
            seq,
            EventPayload::User { tag: seq },
        )
    }

    #[test]
    fn pops_in_ascending_time_order() {
        let mut es = BinaryHeapEventSet::new();
        es.push(event(3.0, Phase::Arrival, 2));
        es.push(event(1.0, Phase::Arrival, 0));
        es.push(event(2.0, Phase::Arrival, 1));

        assert_eq!(es.pop().unwrap().time.as_f64(), 1.0);
        assert_eq!(es.pop().unwrap().time.as_f64(), 2.0);
        assert_eq!(es.pop().unwrap().time.as_f64(), 3.0);
        assert!(es.pop().is_none());
    }

    #[test]
    fn same_time_phase_ordering() {
        let mut es = BinaryHeapEventSet::new();
        es.push(event(5.0, Phase::Arrival, 1));
        es.push(event(5.0, Phase::Interrupt, 0));

        // Interrupt fires before Arrival at the same time
        assert_eq!(es.pop().unwrap().phase, Phase::Interrupt);
        assert_eq!(es.pop().unwrap().phase, Phase::Arrival);
    }

    #[test]
    fn same_time_same_phase_fifo() {
        let mut es = BinaryHeapEventSet::new();
        es.push(event(5.0, Phase::Arrival, 2));
        es.push(event(5.0, Phase::Arrival, 0));
        es.push(event(5.0, Phase::Arrival, 1));

        assert_eq!(es.pop().unwrap().seq, 0);
        assert_eq!(es.pop().unwrap().seq, 1);
        assert_eq!(es.pop().unwrap().seq, 2);
    }

    #[test]
    fn len_and_is_empty() {
        let mut es = BinaryHeapEventSet::new();
        assert!(es.is_empty());
        es.push(event(1.0, Phase::Arrival, 0));
        assert_eq!(es.len(), 1);
        es.pop();
        assert!(es.is_empty());
    }

    #[test]
    fn peek_does_not_remove() {
        let mut es = BinaryHeapEventSet::new();
        es.push(event(1.0, Phase::Arrival, 0));
        assert!(es.peek().is_some());
        assert_eq!(es.len(), 1);
    }
}
