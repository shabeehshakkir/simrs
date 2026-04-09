use pyo3::prelude::*;

/// Minimal Python extension module entry point.
///
/// The binding surface is intentionally empty for now; this keeps the
/// feature-gated crate buildable while the planned Python API is added
/// incrementally.
#[pymodule]
fn simrs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__doc__", "Python bindings for simrs.")?;
    Ok(())
}
