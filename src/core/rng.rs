use rand::distributions::Distribution;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Manages deterministic RNG streams for a simulation run.
///
/// Uses `ChaCha8Rng` for platform-independent reproducibility.
///
/// Seed hierarchy:
/// ```text
/// root_seed
///   └── scenario_seed   = hash(root_seed, scenario_id)
///         └── replication_seed = hash(scenario_seed, rep_id)
///               └── process_seed     = hash(rep_seed, process_id)
/// ```
///
/// All forks are deterministic: same IDs → same stream.
pub struct RngManager {
    /// The active RNG for the current context.
    rng: ChaCha8Rng,
    /// Root seed for auditing.
    root_seed: u64,
}

impl RngManager {
    /// Create an `RngManager` from a root seed.
    pub fn seeded(seed: u64) -> Self {
        RngManager { rng: ChaCha8Rng::seed_from_u64(seed), root_seed: seed }
    }

    /// Derive an independent RNG stream for a specific replication.
    pub fn fork_for_replication(&self, rep_id: u64) -> RngManager {
        let derived = mix(self.root_seed, rep_id);
        RngManager::seeded(derived)
    }

    /// Derive an independent RNG stream for a process within a replication.
    pub fn fork_for_process(&self, process_id: u64) -> RngManager {
        let derived = mix(self.root_seed, process_id ^ 0x9e37_79b9_7f4a_7c15);
        RngManager::seeded(derived)
    }

    /// Sample a value from a distribution.
    pub fn sample<D: Distribution<f64>>(&mut self, dist: &D) -> f64 {
        dist.sample(&mut self.rng)
    }

    /// Sample a `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    /// Return the root seed used to create this manager.
    pub fn root_seed(&self) -> u64 {
        self.root_seed
    }
}

/// Simple, reversible hash mix for seed derivation.
///
/// Based on the splitmix64 finalizer — avalanche mixing ensures
/// that small input differences produce large output differences.
#[inline]
fn mix(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_add(b).wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Uniform;

    #[test]
    fn same_seed_same_sequence() {
        let dist = Uniform::new(0.0, 1.0);
        let mut m1 = RngManager::seeded(42);
        let mut m2 = RngManager::seeded(42);
        for _ in 0..100 {
            assert_eq!(m1.sample(&dist).to_bits(), m2.sample(&dist).to_bits());
        }
    }

    #[test]
    fn different_seeds_different_sequences() {
        let dist = Uniform::new(0.0, 1.0);
        let mut m1 = RngManager::seeded(1);
        let mut m2 = RngManager::seeded(2);
        let v1: Vec<u64> = (0..10).map(|_| m1.sample(&dist).to_bits()).collect();
        let v2: Vec<u64> = (0..10).map(|_| m2.sample(&dist).to_bits()).collect();
        assert_ne!(v1, v2);
    }

    #[test]
    fn replication_forks_are_independent() {
        let manager = RngManager::seeded(0);
        let dist = Uniform::new(0.0, 1.0);
        let mut r0 = manager.fork_for_replication(0);
        let mut r1 = manager.fork_for_replication(1);
        let v0: Vec<u64> = (0..10).map(|_| r0.sample(&dist).to_bits()).collect();
        let v1: Vec<u64> = (0..10).map(|_| r1.sample(&dist).to_bits()).collect();
        assert_ne!(v0, v1);
    }

    #[test]
    fn replication_fork_is_deterministic() {
        let m1 = RngManager::seeded(99);
        let m2 = RngManager::seeded(99);
        let dist = Uniform::new(0.0, 1.0);
        let mut r1 = m1.fork_for_replication(5);
        let mut r2 = m2.fork_for_replication(5);
        for _ in 0..20 {
            assert_eq!(r1.sample(&dist).to_bits(), r2.sample(&dist).to_bits());
        }
    }

    #[test]
    fn process_fork_is_deterministic() {
        let m1 = RngManager::seeded(7);
        let m2 = RngManager::seeded(7);
        let dist = Uniform::new(0.0, 1.0);
        let mut p1 = m1.fork_for_process(3);
        let mut p2 = m2.fork_for_process(3);
        for _ in 0..20 {
            assert_eq!(p1.sample(&dist).to_bits(), p2.sample(&dist).to_bits());
        }
    }
}
