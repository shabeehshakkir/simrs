use crate::core::time::SimTime;
use std::{
    cell::Cell,
    future::Future,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

// ─── Process ID ──────────────────────────────────────────────────────────────

/// Unique identifier for a simulation process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessId(pub(crate) u64);

// ─── No-op waker ─────────────────────────────────────────────────────────────
//
// Processes are polled only when the simulation schedules a ProcessWake event.
// There is no async I/O, so the waker never needs to actually schedule anything.

pub(crate) fn noop_waker() -> Waker {
    unsafe fn noop_clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    unsafe fn noop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

// ─── WakeReason ──────────────────────────────────────────────────────────────

/// Why a process was woken from suspension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeReason {
    /// A `timeout()` completed normally.
    Timeout,
    /// A resource was granted.
    ResourceGranted,
    /// A store/container operation completed.
    StoreReady,
    /// The process was interrupted by another.
    Interrupted,
    /// The process was cancelled.
    Cancelled,
}

// ─── WakeSignal ──────────────────────────────────────────────────────────────

/// Shared signal between the simulation engine and a suspended Future.
///
/// When the engine fires a ProcessWake event, it calls `signal(reason)`.
/// The future then reads the reason on its next poll via `take()`.
///
/// Uses `Cell<Option<WakeReason>>` — safe for single-threaded access.
#[derive(Debug)]
pub(crate) struct WakeSignal {
    inner: Cell<Option<WakeReason>>,
}

impl WakeSignal {
    pub(crate) fn new() -> Rc<Self> {
        Rc::new(WakeSignal { inner: Cell::new(None) })
    }

    pub(crate) fn signal(&self, reason: WakeReason) {
        self.inner.set(Some(reason));
    }

    pub(crate) fn take(&self) -> Option<WakeReason> {
        self.inner.take()
    }
}

// ─── Scheduler callback type ─────────────────────────────────────────────────

/// A callable that schedules an event on the simulation.
///
/// `Rc<dyn Fn(...)>` — single-threaded; no `Send + Sync` needed.
pub(crate) type Scheduler =
    Rc<dyn Fn(SimTime, crate::core::event::Phase, crate::core::event::EventPayload) -> u64>;

// ─── ProcessContext ───────────────────────────────────────────────────────────

/// Scheduling handle passed to every process future.
#[derive(Clone)]
pub struct ProcessContext {
    pub(crate) id: ProcessId,
    pub(crate) scheduler: Scheduler,
    /// Shared "current time" cell — updated before each poll.
    pub(crate) now: Rc<Cell<SimTime>>,
    /// Wake signal for the current suspend point — replaced on each `timeout()`.
    pub(crate) wake_signal: Rc<WakeSignal>,
}

impl ProcessContext {
    /// Current simulation time.
    pub fn now(&self) -> SimTime {
        self.now.get()
    }

    /// Suspend until `dt` sim-time units have elapsed.
    pub fn timeout(&self, dt: f64) -> TimeoutFuture {
        let wake_time = self.now() + dt;
        TimeoutFuture { wake_time, scheduled: false, ctx: self.clone() }
    }

    pub(crate) fn schedule(
        &self,
        time: SimTime,
        phase: crate::core::event::Phase,
        payload: crate::core::event::EventPayload,
    ) -> u64 {
        (self.scheduler)(time, phase, payload)
    }
}

// ─── TimeoutFuture ────────────────────────────────────────────────────────────

pub struct TimeoutFuture {
    wake_time: SimTime,
    scheduled: bool,
    ctx: ProcessContext,
}

impl Unpin for TimeoutFuture {}

impl Future for TimeoutFuture {
    type Output = WakeReason;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<WakeReason> {
        if !self.scheduled {
            self.scheduled = true;
            let process_id = self.ctx.id.0;
            self.ctx.schedule(
                self.wake_time,
                crate::core::event::Phase::Resume,
                crate::core::event::EventPayload::ProcessWake { process_id },
            );
            return Poll::Pending;
        }

        match self.ctx.wake_signal.take() {
            Some(reason) => Poll::Ready(reason),
            None => Poll::Pending,
        }
    }
}

// ─── ProcessEntry ─────────────────────────────────────────────────────────────

pub(crate) struct ProcessEntry {
    pub name: String,
    pub future: Pin<Box<dyn Future<Output = ()>>>,
    pub wake_signal: Rc<WakeSignal>,
    pub context_now: Rc<Cell<SimTime>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wake_signal_roundtrip() {
        let signal = WakeSignal::new();
        signal.signal(WakeReason::Timeout);
        assert_eq!(signal.take(), Some(WakeReason::Timeout));
        assert_eq!(signal.take(), None);
    }
}
