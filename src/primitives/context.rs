use crate::{
    core::{
        event::{EventPayload, Phase},
        simulation::Simulation,
        time::SimTime,
    },
    primitives::process::{noop_waker, ProcessContext, ProcessEntry, ProcessId, WakeReason, WakeSignal},
};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    future::Future,
    pin::Pin,
    rc::Rc,
    task::{Context as TaskContext, Poll},
};

// ─── Shared scheduler state ───────────────────────────────────────────────────

/// Interior-mutable event-queue interface shared between the simulation and
/// process contexts.
///
/// Wrapping in `Rc<RefCell<...>>` lets process futures call `schedule_at`
/// without needing a mutable reference to `Simulation` at the same time as
/// the futures are borrowed.
struct SchedulerCell {
    /// Pending events to schedule — buffered so we don't hold &mut Simulation
    /// while polling a future.
    pending: Vec<(SimTime, Phase, EventPayload)>,
    next_seq: u64,
}

// ─── ProcessRegistry ─────────────────────────────────────────────────────────

/// Manages running process futures and drives execution when wake events fire.
pub struct ProcessRegistry {
    processes: HashMap<u64, ProcessEntry>,
    next_id: u64,
}

impl ProcessRegistry {
    pub fn new() -> Self {
        ProcessRegistry { processes: HashMap::new(), next_id: 0 }
    }

    pub fn running_count(&self) -> usize {
        self.processes.len()
    }

    pub fn is_alive(&self, id: ProcessId) -> bool {
        self.processes.contains_key(&id.0)
    }

    pub(crate) fn wake_signal(&self, id: ProcessId) -> Option<Rc<WakeSignal>> {
        self.processes.get(&id.0).map(|e| Rc::clone(&e.wake_signal))
    }

    /// Poll the process identified by `id`.
    ///
    /// Returns `true` if still running, `false` if completed.
    pub(crate) fn poll_process(&mut self, id: u64, now: SimTime, reason: WakeReason) -> bool {
        let entry = match self.processes.get_mut(&id) {
            Some(e) => e,
            None => return false,
        };

        entry.context_now.set(now);
        entry.wake_signal.signal(reason);

        let waker = noop_waker();
        let mut cx = TaskContext::from_waker(&waker);

        match entry.future.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {
                self.processes.remove(&id);
                false
            }
            Poll::Pending => true,
        }
    }
}

// ─── SimulationWithProcesses ─────────────────────────────────────────────────

/// A `Simulation` plus process registry — the main entry point for
/// process-based modeling.
pub struct SimulationWithProcesses {
    pub sim: Simulation,
    pub registry: ProcessRegistry,
    /// Buffer of events requested by process futures during a poll.
    /// Flushed into `sim` after each poll to avoid aliasing issues.
    scheduled_buffer: Rc<RefCell<Vec<(SimTime, Phase, EventPayload)>>>,
    next_process_seq: u64,
}

impl SimulationWithProcesses {
    pub fn seeded(seed: u64) -> Self {
        SimulationWithProcesses {
            sim: Simulation::seeded(seed),
            registry: ProcessRegistry::new(),
            scheduled_buffer: Rc::new(RefCell::new(Vec::new())),
            next_process_seq: 0,
        }
    }

    pub fn now(&self) -> SimTime {
        self.sim.now()
    }

    /// Register a new process future.
    ///
    /// The process receives a `ProcessContext` for scheduling delays.
    /// It will be polled for the first time at the current simulation time.
    pub fn process<F>(&mut self, name: impl Into<String>, make_future: F) -> ProcessId
    where
        F: FnOnce(ProcessContext) -> Pin<Box<dyn Future<Output = ()>>>,
    {
        let id_u64 = self.registry.next_id;
        self.registry.next_id += 1;
        let process_id = ProcessId(id_u64);

        let wake_signal = WakeSignal::new();
        let context_now = Rc::new(Cell::new(self.sim.now()));
        let buffer = Rc::clone(&self.scheduled_buffer);

        let scheduler: crate::primitives::process::Scheduler =
            Rc::new(move |time, phase, payload| {
                buffer.borrow_mut().push((time, phase, payload));
                0 // seq assigned when flushed
            });

        let ctx = ProcessContext {
            id: process_id,
            scheduler,
            now: Rc::clone(&context_now),
            wake_signal: Rc::clone(&wake_signal),
        };

        let future = make_future(ctx);

        self.registry.processes.insert(
            id_u64,
            ProcessEntry { name: name.into(), future, wake_signal, context_now },
        );

        // Schedule the initial poll at the current time
        self.sim.schedule_at(
            self.sim.now(),
            Phase::Resume,
            EventPayload::ProcessWake { process_id: id_u64 },
        );

        process_id
    }

    /// Flush any events buffered by process futures into the simulation.
    pub(crate) fn flush_buffer(&mut self) {
        let pending: Vec<_> = self.scheduled_buffer.borrow_mut().drain(..).collect();
        for (time, phase, payload) in pending {
            self.sim.schedule_at(time, phase, payload);
        }
    }

    /// Advance by one event.
    ///
    /// Returns `true` if an event was processed, `false` if the event set
    /// is empty (or the time limit was reached).
    pub fn step(&mut self) -> bool {
        let event = match self.sim.step() {
            Some(e) => e,
            None => return false,
        };

        match &event.payload {
            EventPayload::ProcessWake { process_id } => {
                let pid = *process_id;
                let now = self.sim.now();
                let reason = match event.phase {
                    Phase::Interrupt => WakeReason::Interrupted,
                    Phase::Release => WakeReason::ResourceGranted,
                    _ => WakeReason::Timeout,
                };
                self.registry.poll_process(pid, now, reason);
                self.flush_buffer();
            }
            _ => {}
        }

        true
    }

    /// Run until no events remain.
    pub fn run(&mut self) {
        while self.step() {}
    }

    /// Run until the simulation clock reaches `until`.
    pub fn run_until(&mut self, until: f64) {
        let limit = SimTime::from_f64(until);
        loop {
            match self.sim.pending_events_peek() {
                Some(t) if t > limit => break,
                None => break,
                _ => {}
            }
            if !self.step() {
                break;
            }
        }
    }

    pub fn running_processes(&self) -> usize {
        self.registry.running_count()
    }

    pub fn rng_mut(&mut self) -> &mut crate::core::rng::RngManager {
        self.sim.rng_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn single_process_timeout_sequence() {
        let log = Rc::new(RefCell::new(Vec::<f64>::new()));
        let log2 = Rc::clone(&log);

        let mut sim = SimulationWithProcesses::seeded(0);
        sim.process("gen", move |ctx| {
            let log = Rc::clone(&log2);
            Box::pin(async move {
                log.borrow_mut().push(ctx.now().as_f64()); // t=0
                ctx.timeout(5.0).await;
                log.borrow_mut().push(ctx.now().as_f64()); // t=5
                ctx.timeout(3.0).await;
                log.borrow_mut().push(ctx.now().as_f64()); // t=8
            })
        });

        sim.run();

        let times = log.borrow().clone();
        assert_eq!(times, vec![0.0, 5.0, 8.0]);
    }

    #[test]
    fn two_processes_interleave_deterministically() {
        let log = Rc::new(RefCell::new(Vec::<(u8, f64)>::new()));
        let log_a = Rc::clone(&log);
        let log_b = Rc::clone(&log);

        let mut sim = SimulationWithProcesses::seeded(0);

        sim.process("a", move |ctx| {
            let log = Rc::clone(&log_a);
            Box::pin(async move {
                ctx.timeout(2.0).await;
                log.borrow_mut().push((0, ctx.now().as_f64()));
                ctx.timeout(4.0).await;
                log.borrow_mut().push((0, ctx.now().as_f64()));
            })
        });

        sim.process("b", move |ctx| {
            let log = Rc::clone(&log_b);
            Box::pin(async move {
                ctx.timeout(3.0).await;
                log.borrow_mut().push((1, ctx.now().as_f64()));
                ctx.timeout(1.0).await;
                log.borrow_mut().push((1, ctx.now().as_f64()));
            })
        });

        sim.run();

        let events = log.borrow().clone();
        // a wakes at t=2, b wakes at t=3, b wakes again at t=4, a wakes at t=6
        assert_eq!(events, vec![(0, 2.0), (1, 3.0), (1, 4.0), (0, 6.0)]);
    }

    #[test]
    fn process_completes_and_is_removed() {
        let mut sim = SimulationWithProcesses::seeded(0);
        sim.process("one-shot", |ctx| {
            Box::pin(async move {
                ctx.timeout(1.0).await;
            })
        });
        assert_eq!(sim.running_processes(), 1);
        sim.run();
        assert_eq!(sim.running_processes(), 0);
    }
}
