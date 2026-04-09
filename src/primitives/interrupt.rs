use crate::{
    core::event::{EventPayload, Phase},
    primitives::{
        context::SimulationWithProcesses,
        process::{ProcessContext, ProcessId, WakeReason},
    },
};

/// Send an interrupt to another process.
///
/// The target process will be woken at the current simulation time with
/// `WakeReason::Interrupted`. If the target is not currently suspended,
/// the interrupt is silently ignored (the process may have already completed).
pub fn interrupt(ctx: &ProcessContext, target: ProcessId) {
    ctx.schedule(
        ctx.now(),
        Phase::Interrupt,
        EventPayload::ProcessWake { process_id: target.0 },
    );
}

impl SimulationWithProcesses {
    /// Convenience method — interrupt a process by ID.
    ///
    /// Schedules an interrupt event at the current simulation time.
    pub fn interrupt_process(&mut self, target: ProcessId) {
        self.sim.schedule_at(
            self.sim.now(),
            Phase::Interrupt,
            EventPayload::ProcessWake { process_id: target.0 },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{context::SimulationWithProcesses, process::WakeReason};
    use std::{cell::Cell, rc::Rc};

    #[test]
    fn process_can_be_interrupted_during_timeout() {
        // Process A waits for 10 time units.
        // Process B interrupts A after 3 time units.
        // A should wake at t=3 with WakeReason::Interrupted.

        let wake_time = Rc::new(Cell::new(f64::NAN));
        let wake_reason = Rc::new(Cell::new(WakeReason::Timeout));

        let wt = Rc::clone(&wake_time);
        let wr = Rc::clone(&wake_reason);

        let mut sim = SimulationWithProcesses::seeded(0);

        let handle_a = sim.process("a", move |ctx| {
            let wt = Rc::clone(&wt);
            let wr = Rc::clone(&wr);
            Box::pin(async move {
                let reason = ctx.timeout(10.0).await;
                wt.set(ctx.now().as_f64());
                wr.set(reason);
            })
        });

        sim.process("b", move |ctx| {
            let handle = handle_a;
            Box::pin(async move {
                ctx.timeout(3.0).await;
                interrupt(&ctx, handle);
            })
        });

        sim.run();

        // A was interrupted at t=3, not t=10
        assert_eq!(wake_time.get(), 3.0);
        assert_eq!(wake_reason.get(), WakeReason::Interrupted);
    }
}
