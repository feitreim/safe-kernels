//! Vector sum (reduction) — correctness check.
//!
//! The grid-stride kernel is defined once in `src/lib.rs` (shared with the
//! benchmark in `src/bin/bench.rs`); this binary launches it via `reduce` and
//! checks the result against an f64 reference.
//!
//! Run on a GPU (via Modal):  ./run.sh vecsum

use bench_util::uniform_vec;
use cuda_core::{CudaContext, DeviceBuffer};
use nalgebra::DVector;
use vecsum::kernels::{self, M, N};
use vecsum::{GRID, reduce};

fn main() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();

    let a_host = uniform_vec(N, 1);
    let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();

    let module = kernels::from_module(
        ctx.load_module_from_file("vecsum.ptx")
            .expect("load vecsum.ptx"),
    )
    .expect("wrap loaded module");

    let gpu = reduce(&stream, &module, &a).expect("reduction");

    // Accuracy check against an f64 "truth". The kernel sums in f32 through a
    // bounded-depth reduction — each thread grid-strides a short sequential
    // chain, a 256-wide block tree combines those, then the GRID block partials
    // are atomic-added into one accumulator — so the forward error is at most
    // ~depth * eps * ||x||_1. With inputs in [-1, 1) the true sum sits near zero
    // after heavy cancellation, so this gate is loose in relative terms but
    // still catches any gross algorithmic bug. It cannot be tightened to a
    // bit-exact match: the atomic adds accumulate in nondeterministic order.
    let truth: f64 = DVector::from_iterator(N, a_host.iter().map(|&x| x as f64)).sum();
    let l1: f64 = a_host.iter().map(|&x| (x as f64).abs()).sum();

    let per_thread = N.div_ceil(GRID * M); // grid-stride chain length per thread
    let depth = per_thread + M.ilog2() as usize + GRID; // chain + block tree + atomic sum
    let tol = depth as f64 * f32::EPSILON as f64 * l1;

    let err = (gpu as f64 - truth).abs();
    if err <= tol {
        println!(
            "✓ vecsum over {N} elements: gpu = {gpu}, truth = {truth:.4}, |err| = {err:.4e} ≤ tol = {tol:.4e}"
        );
    } else {
        eprintln!(
            "✗ vecsum wrong: gpu = {gpu}, truth = {truth:.4}, |err| = {err:.4e} > tol = {tol:.4e}"
        );
        std::process::exit(1);
    }
}
