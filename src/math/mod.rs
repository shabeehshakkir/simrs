//! Mathematical foundations and OR-facing analytical helpers for `simrs`.
//!
//! Design conventions for this layer:
//! - closed-form formulas and sample estimators live here,
//! - functions return `Option<T>` when invalid inputs make the quantity
//!   undefined under the current lightweight API,
//! - exact formulas, statistical estimators, and heuristic procedures are kept
//!   in separate submodules,
//! - tests should prefer closed-form identities and consistency checks.

pub mod doe;
pub mod optimization;
pub mod queueing;
pub mod ranking_selection;
pub mod risk;
pub mod statistics;
