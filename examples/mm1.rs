//! M/M/1 queue simulation example with warm-up deletion and replications.
//!
//! Usage:
//!   cargo run --example mm1

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Exp};
use simrs::{
    math::queueing::mm1_metrics,
    primitives::{context::SimulationWithProcesses, store::Store},
};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone)]
struct Customer {
    arrival_time: f64,
}

fn run_mm1(
    lambda: f64,
    mu: f64,
    run_length: f64,
    warmup_time: f64,
    seed: u64,
) -> (u64, f64, f64) {
    let completed = Rc::new(RefCell::new(0u64));
    let total_wait = Rc::new(RefCell::new(0.0_f64));
    let total_sojourn = Rc::new(RefCell::new(0.0_f64));

    let queue: Store<Customer> = Store::unbounded();
    let mut sim = SimulationWithProcesses::seeded(seed);

    let q_gen = queue.clone();
    sim.process("generator", move |ctx| {
        let q = q_gen.clone();
        Box::pin(async move {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xDEAD_BEEF);
            let interarrival = Exp::new(lambda).unwrap();
            loop {
                let dt = interarrival.sample(&mut rng);
                ctx.timeout(dt).await;
                q.put(
                    &ctx,
                    Customer {
                        arrival_time: ctx.now().as_f64(),
                    },
                )
                .await;
            }
        })
    });

    let q_srv = queue.clone();
    let co = Rc::clone(&completed);
    let tw = Rc::clone(&total_wait);
    let ts = Rc::clone(&total_sojourn);
    sim.process("server", move |ctx| {
        let q = q_srv.clone();
        Box::pin(async move {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xCAFE_BABE);
            let service = Exp::new(mu).unwrap();
            loop {
                let customer = q.get(&ctx).await;
                let service_start = ctx.now().as_f64();
                let wait = service_start - customer.arrival_time;

                let svc = service.sample(&mut rng);
                ctx.timeout(svc).await;
                let departure = ctx.now().as_f64();

                if departure >= warmup_time {
                    *tw.borrow_mut() += wait;
                    *ts.borrow_mut() += departure - customer.arrival_time;
                    *co.borrow_mut() += 1;
                }
            }
        })
    });

    sim.run_until(run_length);

    let n = *completed.borrow();
    let mean_w = if n > 0 {
        *total_wait.borrow() / n as f64
    } else {
        0.0
    };
    let mean_s = if n > 0 {
        *total_sojourn.borrow() / n as f64
    } else {
        0.0
    };
    (n, mean_w, mean_s)
}

fn main() {
    let lambda = 0.9_f64;
    let mu = 1.0_f64;
    let rho = lambda / mu;
    let run_length = 20_000.0;
    let warmup_time = 5_000.0;
    let replications = 8u64;

    let theory = mm1_metrics(lambda, mu).expect("stable M/M/1");

    let mut total_completed = 0u64;
    let mut mean_waits = Vec::new();
    let mut mean_sojourns = Vec::new();

    for rep in 0..replications {
        let seed = 42 + rep;
        let (n, mean_w, mean_s) = run_mm1(lambda, mu, run_length, warmup_time, seed);
        total_completed += n;
        mean_waits.push(mean_w);
        mean_sojourns.push(mean_s);
    }

    let avg_wait = mean_waits.iter().sum::<f64>() / mean_waits.len() as f64;
    let avg_sojourn = mean_sojourns.iter().sum::<f64>() / mean_sojourns.len() as f64;
    let rel_err_wait = ((avg_wait - theory.mean_waiting_time_in_queue) / theory.mean_waiting_time_in_queue).abs();
    let rel_err_sojourn = ((avg_sojourn - theory.mean_time_in_system) / theory.mean_time_in_system).abs();

    println!("M/M/1 Queue — λ={lambda:.2}, μ={mu:.2}, ρ={rho:.2}");
    println!("  Run length         : {run_length:.0}");
    println!("  Warm-up deleted    : {warmup_time:.0}");
    println!("  Replications       : {replications}");
    println!("  Customers analyzed : {total_completed}");
    println!(
        "  Mean wait  (sim)   : {avg_wait:.4}  (theory: {:.4})",
        theory.mean_waiting_time_in_queue
    );
    println!(
        "  Mean sojourn (sim) : {avg_sojourn:.4}  (theory: {:.4})",
        theory.mean_time_in_system
    );
    println!("  Relative error wait: {:.2}%", rel_err_wait * 100.0);
    println!("  Relative error sys : {:.2}%", rel_err_sojourn * 100.0);

    if rel_err_wait < 0.10 && rel_err_sojourn < 0.10 {
        println!("  [OK] Warmed-up replication averages are within 10% of theory");
    } else {
        println!("  [WARN] Replication averages are still more than 10% from theory");
    }
}
