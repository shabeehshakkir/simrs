//! Warehouse scenario simulation with replicated comparisons.
//!
//! Models a simple warehouse with:
//! - inbound replenishment into an inventory container,
//! - outbound order arrivals,
//! - picker workers,
//! - packer workers.
//!
//! Usage:
//!   cargo run --example warehouse_scenarios

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Exp, Uniform};
use simrs::{
    or::monitors::{CounterMonitor, TallyMonitor, WarehouseKpis},
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
}

#[derive(Clone, Debug)]
struct PickedOrder {
    arrival_time: f64,
    pick_ready_time: f64,
}

#[derive(Clone, Debug)]
struct WarehouseScenario {
    name: &'static str,
    order_rate: f64,
    inbound_rate: f64,
    replenishment_units: f64,
    initial_inventory: f64,
    pickers: usize,
    packers: usize,
    pick_time_per_unit: f64,
    pack_time_per_order: f64,
}

#[derive(Clone, Debug)]
struct ScenarioSummary {
    name: &'static str,
    mean_completed_orders: f64,
    mean_cycle_time: f64,
    mean_pick_wait: f64,
    mean_pack_wait: f64,
    mean_inventory_wait: f64,
}

fn run_scenario(
    scenario: &WarehouseScenario,
    run_length: f64,
    warmup_time: f64,
    seed: u64,
) -> WarehouseKpis {
    let order_queue: Store<Order> = Store::unbounded();
    let pack_queue: Store<PickedOrder> = Store::unbounded();
    let inventory = Container::new(1_000_000.0);

    let completed_orders = Rc::new(RefCell::new(CounterMonitor::new("completed_orders")));
    let cycle_total = Rc::new(RefCell::new(TallyMonitor::new("cycle_time")));
    let pick_wait_total = Rc::new(RefCell::new(TallyMonitor::new("pick_wait")));
    let pack_wait_total = Rc::new(RefCell::new(TallyMonitor::new("pack_wait")));
    let inventory_wait_total = Rc::new(RefCell::new(TallyMonitor::new("inventory_wait")));

    let mut sim = SimulationWithProcesses::seeded(seed);

    if scenario.initial_inventory > 0.0 {
        let inv = inventory.clone();
        let initial_inventory = scenario.initial_inventory;
        sim.process("initial-inventory", move |ctx| {
            let inv = inv.clone();
            Box::pin(async move {
                inv.put(&ctx, initial_inventory).await;
            })
        });
    }

    let q_orders = order_queue.clone();
    let order_rate = scenario.order_rate;
    sim.process("order-generator", move |ctx| {
        let q = q_orders.clone();
        Box::pin(async move {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x0A0D_0001);
            let interarrival = Exp::new(order_rate).unwrap();
            let units = Uniform::new_inclusive(1u32, 4u32);
            loop {
                let dt = interarrival.sample(&mut rng);
                ctx.timeout(dt).await;
                q.put(
                    &ctx,
                    Order {
                        arrival_time: ctx.now().as_f64(),
                        units: units.sample(&mut rng),
                    },
                )
                .await;
            }
        })
    });

    let inv_in = inventory.clone();
    let inbound_rate = scenario.inbound_rate;
    let replenishment_units = scenario.replenishment_units;
    sim.process("replenishment", move |ctx| {
        let inv = inv_in.clone();
        Box::pin(async move {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x1AB0_0002);
            let interarrival = Exp::new(inbound_rate).unwrap();
            loop {
                let dt = interarrival.sample(&mut rng);
                ctx.timeout(dt).await;
                inv.put(&ctx, replenishment_units).await;
            }
        })
    });

    for i in 0..scenario.pickers {
        let q = order_queue.clone();
        let pack_q = pack_queue.clone();
        let inv = inventory.clone();
        let pick_time_per_unit = scenario.pick_time_per_unit;
        let inventory_wait_total = Rc::clone(&inventory_wait_total);
        let pick_wait_total = Rc::clone(&pick_wait_total);
        sim.process(format!("picker-{i}"), move |ctx| {
            let q = q.clone();
            let pack_q = pack_q.clone();
            let inv = inv.clone();
            let inventory_wait_total = Rc::clone(&inventory_wait_total);
            let pick_wait_total = Rc::clone(&pick_wait_total);
            Box::pin(async move {
                loop {
                    let order = q.get(&ctx).await;
                    let pick_start = ctx.now().as_f64();
                    let wait_before_pick = pick_start - order.arrival_time;

                    let inventory_wait_start = ctx.now().as_f64();
                    inv.get(&ctx, order.units as f64).await;
                    let inventory_wait = ctx.now().as_f64() - inventory_wait_start;

                    ctx.timeout(order.units as f64 * pick_time_per_unit).await;
                    let pick_ready_time = ctx.now().as_f64();

                    if pick_ready_time >= warmup_time {
                        inventory_wait_total.borrow_mut().record(inventory_wait);
                        pick_wait_total.borrow_mut().record(wait_before_pick);
                    }

                    pack_q
                        .put(
                            &ctx,
                            PickedOrder {
                                arrival_time: order.arrival_time,
                                pick_ready_time,
                            },
                        )
                        .await;
                }
            })
        });
    }

    for i in 0..scenario.packers {
        let q = pack_queue.clone();
        let completed_orders = Rc::clone(&completed_orders);
        let cycle_total = Rc::clone(&cycle_total);
        let pack_wait_total = Rc::clone(&pack_wait_total);
        let pack_time_per_order = scenario.pack_time_per_order;
        sim.process(format!("packer-{i}"), move |ctx| {
            let q = q.clone();
            let completed_orders = Rc::clone(&completed_orders);
            let cycle_total = Rc::clone(&cycle_total);
            let pack_wait_total = Rc::clone(&pack_wait_total);
            Box::pin(async move {
                loop {
                    let order = q.get(&ctx).await;
                    let pack_start = ctx.now().as_f64();
                    let pack_wait = pack_start - order.pick_ready_time;
                    ctx.timeout(pack_time_per_order).await;
                    let depart = ctx.now().as_f64();

                    if depart >= warmup_time {
                        completed_orders.borrow_mut().increment();
                        cycle_total.borrow_mut().record(depart - order.arrival_time);
                        pack_wait_total.borrow_mut().record(pack_wait);
                    }
                }
            })
        });
    }

    sim.run_until(run_length);

    let completed_ref = completed_orders.borrow();
    let cycle_ref = cycle_total.borrow();
    let pick_ref = pick_wait_total.borrow();
    let pack_ref = pack_wait_total.borrow();
    let inventory_ref = inventory_wait_total.borrow();

    WarehouseKpis::from_monitors(
        &completed_ref,
        &cycle_ref,
        &pick_ref,
        &pack_ref,
        &inventory_ref,
    )
}

fn summarize(
    scenario: &WarehouseScenario,
    run_length: f64,
    warmup_time: f64,
    replications: u64,
    base_seed: u64,
) -> ScenarioSummary {
    let mut completed = 0.0;
    let mut cycle = 0.0;
    let mut pick_wait = 0.0;
    let mut pack_wait = 0.0;
    let mut inventory_wait = 0.0;

    for rep in 0..replications {
        let result = run_scenario(scenario, run_length, warmup_time, base_seed + rep);
        completed += result.completed_orders as f64;
        cycle += result.mean_cycle_time;
        pick_wait += result.pick_stage.mean_wait;
        pack_wait += result.pack_stage.mean_wait;
        inventory_wait += result.inventory_stage.mean_wait;
    }

    let reps = replications as f64;
    ScenarioSummary {
        name: scenario.name,
        mean_completed_orders: completed / reps,
        mean_cycle_time: cycle / reps,
        mean_pick_wait: pick_wait / reps,
        mean_pack_wait: pack_wait / reps,
        mean_inventory_wait: inventory_wait / reps,
    }
}

fn main() {
    let run_length = 20_000.0;
    let warmup_time = 5_000.0;
    let replications = 8;
    let base_seed = 1234;

    let baseline = WarehouseScenario {
        name: "baseline",
        order_rate: 0.85,
        inbound_rate: 0.62,
        replenishment_units: 3.2,
        initial_inventory: 8.0,
        pickers: 2,
        packers: 1,
        pick_time_per_unit: 0.9,
        pack_time_per_order: 0.8,
    };
    let extra_picker = WarehouseScenario {
        name: "extra_picker",
        pickers: 3,
        ..baseline.clone()
    };
    let extra_packer = WarehouseScenario {
        name: "extra_packer",
        packers: 2,
        ..baseline.clone()
    };
    let faster_inbound = WarehouseScenario {
        name: "faster_inbound",
        inbound_rate: 0.90,
        replenishment_units: 3.6,
        ..baseline.clone()
    };
    let low_inventory = WarehouseScenario {
        name: "low_inventory",
        initial_inventory: 2.0,
        inbound_rate: 0.45,
        replenishment_units: 2.6,
        ..baseline.clone()
    };

    let scenarios = vec![baseline, extra_picker, extra_packer, faster_inbound, low_inventory];
    let summaries = scenarios
        .iter()
        .map(|scenario| summarize(scenario, run_length, warmup_time, replications, base_seed))
        .collect::<Vec<_>>();

    for s in &summaries {
        println!(
            "{:<15} completed={:8.1} cycle={:7.3} pick_wait={:7.3} pack_wait={:7.3} inventory_wait={:7.3}",
            s.name,
            s.mean_completed_orders,
            s.mean_cycle_time,
            s.mean_pick_wait,
            s.mean_pack_wait,
            s.mean_inventory_wait,
        );
    }

    let baseline = summaries.iter().find(|s| s.name == "baseline").unwrap();
    let extra_picker = summaries.iter().find(|s| s.name == "extra_picker").unwrap();
    let extra_packer = summaries.iter().find(|s| s.name == "extra_packer").unwrap();
    let faster_inbound = summaries.iter().find(|s| s.name == "faster_inbound").unwrap();
    let low_inventory = summaries.iter().find(|s| s.name == "low_inventory").unwrap();

    assert!(extra_picker.mean_cycle_time < baseline.mean_cycle_time);
    assert!(extra_picker.mean_pick_wait < baseline.mean_pick_wait);
    assert!(extra_packer.mean_pack_wait < baseline.mean_pack_wait);
    assert!(faster_inbound.mean_inventory_wait < baseline.mean_inventory_wait);
    assert!(low_inventory.mean_inventory_wait > baseline.mean_inventory_wait);
    assert!(low_inventory.mean_cycle_time > baseline.mean_cycle_time);

    println!("Warehouse scenario checks passed.");
}
