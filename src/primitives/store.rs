use crate::{
    core::event::{EventPayload, Phase},
    primitives::{
        context::SimulationWithProcesses,
        process::{ProcessContext, ProcessId, WakeReason, WakeSignal},
    },
};
use std::{
    cell::RefCell,
    collections::VecDeque,
    future::Future,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll},
};

// ─── Store<T> ─────────────────────────────────────────────────────────────────

/// A bounded or unbounded FIFO store for items of type `T`.
///
/// Processes can `put` items (blocking if full) or `get` items (blocking if empty).
#[derive(Clone)]
pub struct Store<T: Clone> {
    state: Rc<RefCell<StoreState<T>>>,
}

struct StoreState<T> {
    items: VecDeque<T>,
    capacity: usize,
    /// Processes waiting to put (blocked because store is full)
    put_waiters: VecDeque<PutWaiter<T>>,
    /// Processes waiting to get (blocked because store is empty)
    get_waiters: VecDeque<GetWaiter>,
}

struct PutWaiter<T> {
    item: T,
    process_id: ProcessId,
    signal: Rc<WakeSignal>,
}

struct GetWaiter {
    process_id: ProcessId,
    signal: Rc<WakeSignal>,
}

impl<T: Clone + 'static> Store<T> {
    pub fn new(capacity: usize) -> Self {
        Store {
            state: Rc::new(RefCell::new(StoreState {
                items: VecDeque::new(),
                capacity,
                put_waiters: VecDeque::new(),
                get_waiters: VecDeque::new(),
            })),
        }
    }

    /// Unbounded store.
    pub fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

    pub fn len(&self) -> usize {
        self.state.borrow().items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Put an item into the store, blocking if full.
    pub fn put<'a>(&'a self, ctx: &'a ProcessContext, item: T) -> PutFuture<'a, T> {
        PutFuture { store: self, ctx, item: Some(item), registered: false }
    }

    /// Get the next item from the store, blocking if empty.
    pub fn get<'a>(&'a self, ctx: &'a ProcessContext) -> GetFuture<'a, T> {
        GetFuture { store: self, ctx, registered: false }
    }
}

// ─── PutFuture ────────────────────────────────────────────────────────────────

pub struct PutFuture<'a, T: Clone> {
    store: &'a Store<T>,
    ctx: &'a ProcessContext,
    item: Option<T>,
    registered: bool,
}

impl<T: Clone> Unpin for PutFuture<'_, T> {}

impl<'a, T: Clone + 'static> Future for PutFuture<'a, T> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if !this.registered {
            this.registered = true;
            let mut state = this.store.state.borrow_mut();

            if state.items.len() < state.capacity {
                if let Some(getter) = state.get_waiters.pop_front() {
                    state.items.push_back(this.item.take().unwrap());
                    this.ctx.schedule(
                        this.ctx.now(),
                        Phase::Resume,
                        EventPayload::ProcessWake { process_id: getter.process_id.0 },
                    );
                } else {
                    state.items.push_back(this.item.take().unwrap());
                }
                return Poll::Ready(());
            }

            state.put_waiters.push_back(PutWaiter {
                item: this.item.clone().unwrap(),
                process_id: this.ctx.id,
                signal: Rc::clone(&this.ctx.wake_signal),
            });
            return Poll::Pending;
        }

        Poll::Ready(())
    }
}

// ─── GetFuture ────────────────────────────────────────────────────────────────

pub struct GetFuture<'a, T: Clone> {
    store: &'a Store<T>,
    ctx: &'a ProcessContext,
    registered: bool,
}

impl<T: Clone> Unpin for GetFuture<'_, T> {}

impl<'a, T: Clone + 'static> Future for GetFuture<'a, T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<T> {
        if !self.registered {
            self.registered = true;
            let mut state = self.store.state.borrow_mut();

            if let Some(item) = state.items.pop_front() {
                // Wake a put waiter if any were blocked
                if let Some(waiter) = state.put_waiters.pop_front() {
                    state.items.push_back(waiter.item);
                    self.ctx.schedule(
                        self.ctx.now(),
                        Phase::Resume,
                        EventPayload::ProcessWake { process_id: waiter.process_id.0 },
                    );
                }
                return Poll::Ready(item);
            }

            // Store is empty — queue this get
            state.get_waiters.push_back(GetWaiter {
                process_id: self.ctx.id,
                signal: Rc::clone(&self.ctx.wake_signal),
            });
            return Poll::Pending;
        }

        // Re-polled after a wake — item should be waiting
        let mut state = self.store.state.borrow_mut();
        match state.items.pop_front() {
            Some(item) => Poll::Ready(item),
            None => Poll::Pending, // shouldn't happen with correct use
        }
    }
}

// ─── Container ───────────────────────────────────────────────────────────────

/// A numeric-level container (like a tank or buffer).
///
/// Supports `put(amount).await` and `get(amount).await`.
#[derive(Clone)]
pub struct Container {
    state: Rc<RefCell<ContainerState>>,
}

struct ContainerState {
    level: f64,
    capacity: f64,
    put_waiters: VecDeque<ContainerPutWaiter>,
    get_waiters: VecDeque<ContainerGetWaiter>,
}

struct ContainerPutWaiter {
    amount: f64,
    process_id: ProcessId,
    signal: Rc<WakeSignal>,
}

struct ContainerGetWaiter {
    amount: f64,
    process_id: ProcessId,
    signal: Rc<WakeSignal>,
    /// Amount actually delivered — set when woken.
    delivered: Rc<RefCell<Option<f64>>>,
}

impl Container {
    pub fn new(capacity: f64) -> Self {
        Container {
            state: Rc::new(RefCell::new(ContainerState {
                level: 0.0,
                capacity,
                put_waiters: VecDeque::new(),
                get_waiters: VecDeque::new(),
            })),
        }
    }

    pub fn level(&self) -> f64 {
        self.state.borrow().level
    }

    pub fn capacity(&self) -> f64 {
        self.state.borrow().capacity
    }

    /// Put `amount` into the container, blocking if it would overflow.
    pub fn put<'a>(&'a self, ctx: &'a ProcessContext, amount: f64) -> ContainerPutFuture<'a> {
        ContainerPutFuture { container: self, ctx, amount, registered: false }
    }

    /// Get `amount` from the container, blocking until available.
    pub fn get<'a>(&'a self, ctx: &'a ProcessContext, amount: f64) -> ContainerGetFuture<'a> {
        ContainerGetFuture { container: self, ctx, amount, registered: false }
    }
}

pub struct ContainerPutFuture<'a> {
    container: &'a Container,
    ctx: &'a ProcessContext,
    amount: f64,
    registered: bool,
}

impl Unpin for ContainerPutFuture<'_> {}

impl<'a> Future for ContainerPutFuture<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if !self.registered {
            self.registered = true;
            let mut state = self.container.state.borrow_mut();

            if state.level + self.amount <= state.capacity {
                state.level += self.amount;
                // Wake get waiters if any can now be satisfied
                while let Some(waiter) = state.get_waiters.front() {
                    if state.level >= waiter.amount {
                        let waiter = state.get_waiters.pop_front().unwrap();
                        state.level -= waiter.amount;
                        self.ctx.schedule(
                            self.ctx.now(),
                            Phase::Resume,
                            EventPayload::ProcessWake { process_id: waiter.process_id.0 },
                        );
                    } else {
                        break;
                    }
                }
                return Poll::Ready(());
            }

            state.put_waiters.push_back(ContainerPutWaiter {
                amount: self.amount,
                process_id: self.ctx.id,
                signal: Rc::clone(&self.ctx.wake_signal),
            });
            return Poll::Pending;
        }

        Poll::Ready(())
    }
}

pub struct ContainerGetFuture<'a> {
    container: &'a Container,
    ctx: &'a ProcessContext,
    amount: f64,
    registered: bool,
}

impl Unpin for ContainerGetFuture<'_> {}

impl<'a> Future for ContainerGetFuture<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if !self.registered {
            self.registered = true;
            let mut state = self.container.state.borrow_mut();

            if state.level >= self.amount {
                state.level -= self.amount;
                // Wake put waiters
                while let Some(waiter) = state.put_waiters.front() {
                    if state.level + waiter.amount <= state.capacity {
                        let waiter = state.put_waiters.pop_front().unwrap();
                        state.level += waiter.amount;
                        self.ctx.schedule(
                            self.ctx.now(),
                            Phase::Resume,
                            EventPayload::ProcessWake { process_id: waiter.process_id.0 },
                        );
                    } else {
                        break;
                    }
                }
                return Poll::Ready(());
            }

            state.get_waiters.push_back(ContainerGetWaiter {
                amount: self.amount,
                process_id: self.ctx.id,
                signal: Rc::clone(&self.ctx.wake_signal),
                delivered: Rc::new(RefCell::new(None)),
            });
            return Poll::Pending;
        }

        Poll::Ready(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::context::SimulationWithProcesses;
    use std::cell::RefCell;

    #[test]
    fn store_put_then_get() {
        let result = Rc::new(RefCell::new(None::<i32>));
        let result2 = Rc::clone(&result);

        let store: Store<i32> = Store::new(10);
        let s1 = store.clone();
        let s2 = store.clone();

        let mut sim = SimulationWithProcesses::seeded(0);

        sim.process("producer", move |ctx| {
            let s = s1.clone();
            Box::pin(async move {
                s.put(&ctx, 42).await;
            })
        });

        sim.process("consumer", move |ctx| {
            let s = s2.clone();
            let r = Rc::clone(&result2);
            Box::pin(async move {
                let v = s.get(&ctx).await;
                *r.borrow_mut() = Some(v);
            })
        });

        sim.run();
        assert_eq!(*result.borrow(), Some(42));
    }

    #[test]
    fn store_get_blocks_until_put() {
        let log = Rc::new(RefCell::new(Vec::<&str>::new()));
        let la = Rc::clone(&log);
        let lb = Rc::clone(&log);

        let store: Store<i32> = Store::new(1);
        let s1 = store.clone();
        let s2 = store.clone();

        let mut sim = SimulationWithProcesses::seeded(0);

        // consumer tries to get first (store is empty)
        sim.process("consumer", move |ctx| {
            let s = s1.clone();
            let l = Rc::clone(&la);
            Box::pin(async move {
                l.borrow_mut().push("waiting");
                let _v = s.get(&ctx).await;
                l.borrow_mut().push("got");
            })
        });

        // producer puts after a timeout
        sim.process("producer", move |ctx| {
            let s = s2.clone();
            let l = Rc::clone(&lb);
            Box::pin(async move {
                ctx.timeout(2.0).await;
                l.borrow_mut().push("putting");
                s.put(&ctx, 1).await;
            })
        });

        sim.run();
        assert_eq!(*log.borrow(), vec!["waiting", "putting", "got"]);
    }

    #[test]
    fn container_put_and_get() {
        let container = Container::new(100.0);
        let c1 = container.clone();
        let c2 = container.clone();

        let mut sim = SimulationWithProcesses::seeded(0);

        sim.process("filler", move |ctx| {
            let c = c1.clone();
            Box::pin(async move {
                c.put(&ctx, 50.0).await;
            })
        });

        sim.process("drainer", move |ctx| {
            let c = c2.clone();
            Box::pin(async move {
                ctx.timeout(1.0).await;
                c.get(&ctx, 30.0).await;
            })
        });

        sim.run();
        assert!((container.level() - 20.0).abs() < 1e-9);
    }
}
