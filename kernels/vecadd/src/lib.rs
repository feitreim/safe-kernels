//! The `#[kernel]` function inside `#[cuda_module]` is compiled to PTX by the
//! cuda-oxide codegen backend and written to `vecadd.ptx` next to this crate.
//! Shared by `main.rs` (correctness check) and `src/bin/bench.rs` (throughput
//! benchmark) so the kernel is defined exactly once.
//!
//! Each binary loads the PTX file directly (`ctx.load_module_from_file` +
//! `kernels::from_module`) rather than `kernels::load`, which only looks for
//! a PTX artifact embedded in the *currently running executable*. Since the
//! kernel now lives in this lib crate rather than directly in a `main.rs`,
//! the linker drops that embedded artifact as dead weight from `main`/`bench`
//! (nothing in their compiled code references it) — `kernels::load` would
//! fail with `ModuleNotFound`. Loading from the file side-steps that.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}
