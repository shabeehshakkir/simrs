//! Longer, harder warehouse stress simulation with time-varying Poisson flows.
//!
//! Usage:
//!   cargo run --example warehouse_stress

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Exp, Uniform};
use simrs::{
    or::monitors::{CounterMonitor, TallyMonitor},
    primitives::{
        context::SimulationWithProcesses,
        store::{Container, Store},
    },
};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Debug)]
struct Order {
    arrival_time: f64,
    units: u32,
    rush: bool,
}

#[derive(Clone, Debug)]
struct PickedOrder {
    arrival_time: f64,
    pick_ready_time: f64,
    units: u32,
    rush: bool,
}

#[derive(Clone, Debug)]
struct StressScenario {
    name: &'static str,
    initial_inventory: f64,
    regular_rates: [f64; 3],
    rush_rates: [f64; 3],
    inbound_rates: [f64; 3],
    replenishment_units: f64,
    pickers: usize,
    packers: usize,
    pick_time_per_unit: f64,
    pack_time_regular: f64,
    pack_time_rush: f64,
}

#[derive(Clone, Debug)]
struct StressResult {
    completed_orders: u64,
    rush_share: f64,
    mean_cycle_time: f64,
    mean_inventory_wait: f64,
    mean_pack_wait: f64,
    mean_pick_backlog_delay: f64,
}

fn phase_index(now: f64, run_length: f64, warmup_time: f64) -> usize {
    let measured_span = (run_length - warmup_time).max(1.0);
    let shifted = (now - warmup_time).max(0.0);
    let third = measured_span / 3.0;
    if shifted < third {
        0
    } else if shifted < 2.0 * third {
        1
    } else {
        2
    }
}

fn run_stress_scenario(
    scenario: &StressScenario,
    run_length: f64,
    warmup_time: f64,
    seed: u64,
) -> StressResult {
    let order_queue: Store<Order> = Store::unbounded();
    let pack_queue: Store<PickedOrder> = Store::unbounded();
    let inventory = Container::new(1_000_000.0);

    let completed_orders = Rc::new(RefCell::new(CounterMonitor::new("completed_orders")));
    let rush_completed = Rc::new(RefCell::new(CounterMonitor::new("rush_completed")));
    let cycle_total = Rc::new(RefCell::new(TallyMonitor::new("cycle_time")));
    let inventory_wait_total = Rc::new(RefCell::new(TallyMonitor::new("inventory_wait")));
    let pack_wait_total = Rc::new(RefCell::new(TallyMonitor::new("pack_wait")));
    let pick_backlog_delay_total = Rc::new(RefCell::new(TallyMonitor::new("pick_backlog_delay")));

    let mut sim = SimulationWithProcesses::seeded(seed);

    if scenario.initial_inventory > 0.0 {
        let inv = inventory.clone();
        let qty = scenario.initial_inventory;
        sim.process("initial-inventory", move |ctx| {
            let inv = inv.clone();
            Box::pin(async move {
                inv.put(&ctx, qty).await;
            })
        });
    }

    // Regular order stream: time-varying Poisson via piecewise exponential clocks.
    let q_regular = order_queue.clone();
    let regular_rates = scenario.regular_rates;
    sim.process("regular-orders", move |ctx| {
        let q = q_regular.clone();
        Box::pin(async move {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xAA11_0001);
            let units = Uniform::new_inclusive(1u32, 6u32);
            loop {
                let phase = phase_index(ctx.now().as_f64(), run_length, warmup_time);
                let rate = regular_rates[phase];
                let dt = Exp::new(rate).unwrap().sample(&mut rng);
                ctx.timeout(dt).await;
                q.put(
                    &ctx,
                    Order {
                        arrival_time: ctx.now().as_f64(),
                        units: units.sample(&mut rng),
                        rush: false,
                    },
                )
                .await;
            }
        })
    });

    // Rush order stream.
    let q_rush = order_queue.clone();
    let rush_rates = scenario.rush_rates;
    sim.process("rush-orders", move |ctx| {
        let q = q_rush.clone();
        Box::pin(async move {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xBB22_0002);
            let units = Uniform::new_inclusive(1u32, 3u32);
            loop {
                let phase = phase_index(ctx.now().as_f64(), run_length, warmup_time);
                let rate = rush_rates[phase];
                let dt = Exp::new(rate).unwrap().sample(&mut rng);
                ctx.timeout(dt).await;
                q.put(
                    &ctx,
                    Order {
                        arrival_time: ctx.now().as_f64(),
                        units: units.sample(&mut rng),
                        rush: true,
                    },
                )
                .await;
            }
        })
    });

    // Time-varying replenishment stream.
    let inv_in = inventory.clone();
    let inbound_rates = scenario.inbound_rates;
    let replenishment_units = scenario.replenishment_units;
    sim.process("inbound", move |ctx| {
        let inv = inv_in.clone();
        Box::pin(async move {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xCC33_0003);
            loop {
                let phase = phase_index(ctx.now().as_f64(), run_length, warmup_time);
                let rate = inbound_rates[phase];
                let dt = Exp::new(rate).unwrap().sample(&mut rng);
                ctx.timeout(dt).await;
                inv.put(&ctx, replenishment_units).await;
            }
        })
    });

    // Picker workers.
    for i in 0..scenario.pickers {
        let q = order_queue.clone();
        let out = pack_queue.clone();
        let inv = inventory.clone();
        let inventory_wait_total = Rc::clone(&inventory_wait_total);
        let pick_backlog_delay_total = Rc::clone(&pick_backlog_delay_total);
        let pick_time_per_unit = scenario.pick_time_per_unit;
        sim.process(format!("picker-{i}"), move |ctx| {
            let q = q.clone();
            let out = out.clone();
            let inv = inv.clone();
            let inventory_wait_total = Rc::clone(&inventory_wait_total);
            let pick_backlog_delay_total = Rc::clone(&pick_backlog_delay_total);
            Box::pin(async move {
                loop {
                    let order = q.get(&ctx).await;
                    let start_pick = ctx.now().as_f64();
                    let backlog_delay = start_pick - order.arrival_time;

                    let inv_wait_start = ctx.now().as_f64();
                    inv.get(&ctx, order.units as f64).await;
                    let inventory_wait = ctx.now().as_f64() - inv_wait_start;

                    ctx.timeout(order.units as f64 * pick_time_per_unit).await;
                    let pick_ready = ctx.now().as_f64();

                    if pick_ready >= warmup_time {
                        pick_backlog_delay_total.borrow_mut().record(backlog_delay);
                        inventory_wait_total.borrow_mut().record(inventory_wait);
                    }

                    out.put(
                        &ctx,
                        PickedOrder {
                            arrival_time: order.arrival_time,
                            pick_ready_time: pick_ready,
                            units: order.units,
                            rush: order.rush,
                        },
                    )
                    .await;
                }
            })
        });
    }

    // Packers.
    for i in 0..scenario.packers {
        let q = pack_queue.clone();
        let completed_orders = Rc::clone(&completed_orders);
        let rush_completed = Rc::clone(&rush_completed);
        let cycle_total = Rc::clone(&cycle_total);
        let pack_wait_total = Rc::clone(&pack_wait_total);
        let regular_pack = scenario.pack_time_regular;
        let rush_pack = scenario.pack_time_rush;
        sim.process(format!("packer-{i}"), move |ctx| {
            let q = q.clone();
            let completed_orders = Rc::clone(&completed_orders);
            let rush_completed = Rc::clone(&rush_completed);
            let cycle_total = Rc::clone(&cycle_total);
            let pack_wait_total = Rc::clone(&pack_wait_total);
            Box::pin(async move {
                loop {
                    let order = q.get(&ctx).await;
                    let pack_start = ctx.now().as_f64();
                    let pack_wait = pack_start - order.pick_ready_time;
                    let pack_time = if order.rush { rush_pack } else { regular_pack };
                    let _units = order.units;
                    ctx.timeout(pack_time).await;
                    let depart = ctx.now().as_f64();

                    if depart >= warmup_time {
                        completed_orders.borrow_mut().increment();
                        if order.rush {
                            rush_completed.borrow_mut().increment();
                        }
                        cycle_total.borrow_mut().record(depart - order.arrival_time);
                        pack_wait_total.borrow_mut().record(pack_wait);
                    }
                }
            })
        });
    }

    sim.run_until(run_length);

    let completed = completed_orders.borrow().count();
    let denom = completed.max(1) as f64;

    let rush_share = rush_completed.borrow().count() as f64 / denom;
    let cycle = cycle_total.borrow().mean().unwrap_or(0.0);
    let inventory_wait = inventory_wait_total.borrow().mean().unwrap_or(0.0);
    let pack_wait = pack_wait_total.borrow().mean().unwrap_or(0.0);
    let pick_delay = pick_backlog_delay_total.borrow().mean().unwrap_or(0.0);

    StressResult {
        completed_orders: completed,
        rush_share,
        mean_cycle_time: cycle,
        mean_inventory_wait: inventory_wait,
        mean_pack_wait: pack_wait,
        mean_pick_backlog_delay: pick_delay,
    }
}

fn summarize_scenario(
    scenario: &StressScenario,
    run_length: f64,
    warmup_time: f64,
    replications: u64,
    base_seed: u64,
) -> StressResult {
    let mut completed = 0.0;
    let mut rush_share = 0.0;
    let mut cycle = 0.0;
    let mut inventory_wait = 0.0;
    let mut pack_wait = 0.0;
    let mut pick_delay = 0.0;

    for rep in 0..replications {
        let r = run_stress_scenario(scenario, run_length, warmup_time, base_seed + rep);
        completed += r.completed_orders as f64;
        rush_share += r.rush_share;
        cycle += r.mean_cycle_time;
        inventory_wait += r.mean_inventory_wait;
        pack_wait += r.mean_pack_wait;
        pick_delay += r.mean_pick_backlog_delay;
    }

    let reps = replications as f64;
    StressResult {
        completed_orders: (completed / reps).round() as u64,
        rush_share: rush_share / reps,
        mean_cycle_time: cycle / reps,
        mean_inventory_wait: inventory_wait / reps,
        mean_pack_wait: pack_wait / reps,
        mean_pick_backlog_delay: pick_delay / reps,
    }
}

fn main() {
    let run_length = 100_000.0;
    let warmup_time = 20_000.0;
    let replications = 4;
    let base_seed = 9000;

    let balanced = StressScenario {
        name: "balanced",
        initial_inventory: 30.0,
        regular_rates: [0.45, 0.75, 0.55],
        rush_rates: [0.05, 0.18, 0.08],
        inbound_rates: [0.55, 0.78, 0.60],
        replenishment_units: 3.8,
        pickers: 3,
        packers: 2,
        pick_time_per_unit: 0.75,
        pack_time_regular: 0.65,
        pack_time_rush: 0.45,
    };
    let peak_demand = StressScenario {
        name: "peak_demand",
        regular_rates: [0.55, 1.00, 0.70],
        rush_rates: [0.08, 0.30, 0.12],
        ..balanced.clone()
    };
    let tight_inbound = StressScenario {
        name: "tight_inbound",
        inbound_rates: [0.40, 0.50, 0.42],
        replenishment_units: 3.1,
        initial_inventory: 10.0,
        ..balanced.clone()
    };
    let added_capacity = StressScenario {
        name: "added_capacity",
        pickers: 4,
        packers: 3,
        ..balanced.clone()
    };

    let scenarios = [balanced, peak_demand, tight_inbound, added_capacity];
    let summaries = scenarios
        .iter()
        .map(|s| (s.name, summarize_scenario(s, run_length, warmup_time, replications, base_seed)))
        .collect::<Vec<_>>();

    println!("== Warehouse stress test ==");
    println!("run_length={run_length:.0}, warmup={warmup_time:.0}, replications={replications}");
    for (name, s) in &summaries {
        println!(
            "{:<15} completed={:8} cycle={:8.3} pick_delay={:8.3} pack_wait={:7.3} inventory_wait={:7.3} rush_share={:6.3}",
            name,
            s.completed_orders,
            s.mean_cycle_time,
            s.mean_pick_backlog_delay,
            s.mean_pack_wait,
            s.mean_inventory_wait,
            s.rush_share,
        );
    }

    let balanced = &summaries[0].1;
    let peak = &summaries[1].1;
    let tight = &summaries[2].1;
    let added = &summaries[3].1;

    assert!(peak.mean_cycle_time > balanced.mean_cycle_time);
    assert!(peak.mean_pick_backlog_delay > balanced.mean_pick_backlog_delay);
    assert!(tight.mean_inventory_wait > balanced.mean_inventory_wait);
    assert!(tight.completed_orders < balanced.completed_orders);
    assert!(added.mean_cycle_time < balanced.mean_cycle_time);
    assert!(added.mean_pick_backlog_delay < balanced.mean_pick_backlog_delay);

    println!("Stress scenario checks passed.");
}
