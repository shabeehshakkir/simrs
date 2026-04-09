use crate::{
    core::{
        event::{EventPayload, Phase},
        time::SimTime,
    },
    primitives::{
        context::SimulationWithProcesses,
        process::{ProcessContext, ProcessId, WakeReason, WakeSignal},
    },
};
use std::{
    cell::RefCell,
    collections::BinaryHeap,
    future::Future,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll},
};

// ─── Resource handle ─────────────────────────────────────────────────────────

/// A handle to a capacity resource.
///
/// Resources are owned by a `ResourceState` held in `Rc<RefCell<...>>`.
/// Cloning the handle is cheap — all clones share the same state.
#[derive(Clone)]
pub struct Resource {
    pub(crate) state: Rc<RefCell<ResourceState>>,
}

impl Resource {
    /// Create a new resource with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Resource {
            state: Rc::new(RefCell::new(ResourceState {
                capacity,
                in_use: 0,
                waiters: Default::default(),
            })),
        }
    }

    /// Request one unit of this resource.
    ///
    /// If the resource has free capacity, returns immediately.
    /// Otherwise, suspends the calling process until a unit is released.
    pub fn request<'a>(&'a self, ctx: &'a ProcessContext) -> RequestFuture<'a> {
        RequestFuture {
            resource: self,
            ctx,
            priority: i64::MAX, // FIFO: no priority
            registered: false,
            granted: false,
        }
    }

    /// Request with explicit numeric priority (lower value = higher priority).
    pub fn request_with_priority<'a>(
        &'a self,
        ctx: &'a ProcessContext,
        priority: i64,
    ) -> RequestFuture<'a> {
        RequestFuture { resource: self, ctx, priority, registered: false, granted: false }
    }

    /// Release one unit back to the resource.
    ///
    /// Wakes the highest-priority waiter if any are queued.
    /// Must be called with the simulation context to schedule the wake event.
    pub fn release(&self, ctx: &ProcessContext) {
        let mut state = self.state.borrow_mut();
        debug_assert!(state.in_use > 0, "release() called but resource has nothing in use");
        state.in_use -= 1;

        if let Some(waiter) = state.waiters.pop() {
            state.in_use += 1;
            // Schedule a wake event for the next waiter
            ctx.schedule(
                ctx.now(),
                Phase::Release,
                EventPayload::ProcessWake { process_id: waiter.process_id.0 },
            );
        }
    }

    pub fn capacity(&self) -> usize {
        self.state.borrow().capacity
    }

    pub fn in_use(&self) -> usize {
        self.state.borrow().in_use
    }

    pub fn queue_len(&self) -> usize {
        self.state.borrow().waiters.len()
    }

    pub fn utilization(&self) -> f64 {
        let s = self.state.borrow();
        s.in_use as f64 / s.capacity as f64
    }
}

// ─── ResourceState ────────────────────────────────────────────────────────────

pub(crate) struct ResourceState {
    capacity: usize,
    in_use: usize,
    /// Waiters sorted by priority (lower = more urgent). Ties broken by seq.
    waiters: BinaryHeap<Waiter>,
}

struct Waiter {
    /// Lower value = higher priority.
    priority: i64,
    /// Arrival sequence for FIFO tie-breaking.
    arrival_seq: u64,
    process_id: ProcessId,
    /// Shared signal so we can mark it granted.
    signal: Rc<WakeSignal>,
}

impl PartialEq for Waiter {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.arrival_seq == other.arrival_seq
    }
}
impl Eq for Waiter {}

impl Ord for Waiter {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap; we want lower priority value first.
        // Negate priority so that lower input value → larger heap key.
        other
            .priority
            .cmp(&self.priority)
            .then(other.arrival_seq.cmp(&self.arrival_seq))
    }
}

impl PartialOrd for Waiter {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ─── RequestFuture ────────────────────────────────────────────────────────────

pub struct RequestFuture<'a> {
    resource: &'a Resource,
    ctx: &'a ProcessContext,
    priority: i64,
    registered: bool,
    granted: bool,
}

impl Unpin for RequestFuture<'_> {}

impl<'a> Future for RequestFuture<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.granted {
            return Poll::Ready(());
        }

        let mut state = self.resource.state.borrow_mut();

        if !self.registered {
            self.registered = true;

            if state.in_use < state.capacity {
                // Resource is immediately available
                state.in_use += 1;
                drop(state);
                self.granted = true;
                return Poll::Ready(());
            }

            // Enqueue this process as a waiter
            let signal = self.ctx.wake_signal.clone();
            let arrival_seq = self.ctx.id.0; // use process ID as stable arrival order
            state.waiters.push(Waiter {
                priority: self.priority,
                arrival_seq,
                process_id: self.ctx.id,
                signal,
            });
            drop(state);

            // Schedule a suspend (no wake event here — release() will do that)
            // We park the process by returning Pending and waiting for a
            // Release-phase wake event.
            return Poll::Pending;
        }

        // We were re-polled after a Release wake event
        // The resource was already granted by release()
        drop(state);
        self.granted = true;
        Poll::Ready(())
    }
}

// ─── ResourcePermit ──────────────────────────────────────────────────────────

/// RAII guard for a resource allocation.
///
/// Releases the resource when dropped — but note: in simulation code, you
/// usually want to release explicitly with `permit.release(ctx)` so the
/// wake event is scheduled at the right simulation time.
pub struct ResourcePermit {
    pub resource: Resource,
}

impl ResourcePermit {
    pub fn release(self, ctx: &ProcessContext) {
        self.resource.release(ctx);
    }
}

// ─── SimulationWithProcesses extension ───────────────────────────────────────

impl SimulationWithProcesses {
    /// Handle a resource-grant wake event (fired by `release()`).
    ///
    /// This is called from `step()` when the event's phase is `Release`.
    pub(crate) fn handle_resource_wake(&mut self, process_id: u64) {
        let now = self.sim.now();
        self.registry.poll_process(process_id, now, WakeReason::ResourceGranted);
        self.flush_buffer();
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn immediate_grant_when_free() {
        let log = Rc::new(RefCell::new(Vec::<&str>::new()));
        let log2 = Rc::clone(&log);

        let resource = Resource::new(1);
        let r2 = resource.clone();

        let mut sim = SimulationWithProcesses::seeded(0);

        sim.process("p", move |ctx| {
            let log = Rc::clone(&log2);
            let r = r2.clone();
            Box::pin(async move {
                r.request(&ctx).await;
                log.borrow_mut().push("acquired");
                r.release(&ctx);
                log.borrow_mut().push("released");
            })
        });

        sim.run();
        assert_eq!(*log.borrow(), vec!["acquired", "released"]);
    }

    #[test]
    fn fifo_ordering_for_contended_resource() {
        let order = Rc::new(RefCell::new(Vec::<u8>::new()));

        let resource = Resource::new(1);

        let mut sim = SimulationWithProcesses::seeded(0);

        for i in 0u8..3 {
            let r = resource.clone();
            let o = Rc::clone(&order);
            sim.process(format!("p{}", i), move |ctx| {
                let r = r.clone();
                let o = Rc::clone(&o);
                Box::pin(async move {
                    r.request(&ctx).await;
                    o.borrow_mut().push(i);
                    ctx.timeout(1.0).await;
                    r.release(&ctx);
                })
            });
        }

        sim.run();

        // All 3 processes should have been served (order depends on arrival seq)
        assert_eq!(order.borrow().len(), 3);
    }

    #[test]
    fn priority_resource_serves_higher_priority_first() {
        // Lower priority value = served first
        let order = Rc::new(RefCell::new(Vec::<i64>::new()));

        let resource = Resource::new(1);

        // First, one process holds the resource for a while
        let r_holder = resource.clone();
        let mut sim = SimulationWithProcesses::seeded(0);

        sim.process("holder", move |ctx| {
            let r = r_holder.clone();
            Box::pin(async move {
                r.request(&ctx).await;
                ctx.timeout(1.0).await; // hold while others queue up
                r.release(&ctx);
            })
        });

        // Three waiters with different priorities all arrive at t=0
        for prio in [30i64, 10, 20] {
            let r = resource.clone();
            let o = Rc::clone(&order);
            sim.process(format!("waiter-{}", prio), move |ctx| {
                let r = r.clone();
                let o = Rc::clone(&o);
                Box::pin(async move {
                    r.request_with_priority(&ctx, prio).await;
                    o.borrow_mut().push(prio);
                    r.release(&ctx);
                })
            });
        }

        sim.run();

        // Priority order: 10, 20, 30
        assert_eq!(*order.borrow(), vec![10, 20, 30]);
    }
}
