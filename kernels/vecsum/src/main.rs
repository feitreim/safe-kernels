//! Vector sum (reduction) — correctness check.
//!
//! The kernel is defined once in `src/lib.rs` (shared with the benchmark in
//! `src/bin/bench.rs`); this binary just launches it and checks the result.
//!
//! Run on a GPU (via Modal):  ./run.sh vecsum

use bench_util::uniform_vec;
use cuda_core::{CudaContext, DeviceBuffer};
use nalgebra::DVector;
use vecsum::kernels::{self, M};
use vecsum::reduce;

/// Reduce one `M`-element block exactly as the `vecsum` kernel does: interleaved
/// pairwise addressing (`TILE[index] += TILE[index + s]` for `index = 2*s*tid`).
/// Runs in `f32`, so the block sum is bit-identical to the device's.
fn block_reduce(block: &[f32]) -> f32 {
    let bdim = block.len();
    let mut tile = [0.0f32; M];
    tile[..bdim].copy_from_slice(block);

    let mut s = 1;
    while s < bdim {
        let mut index = 0;
        while index < bdim {
            tile[index] += tile[index + s];
            index += 2 * s;
        }
        s *= 2;
    }
    tile[0]
}

/// Host replica of the whole recursive reduction: reduce in `M`-sized blocks,
/// then reduce those block sums the same way, until one value remains — exactly
/// the tree of launches the device runs. Every add is the same `f32` add in the
/// same order, so this is bit-identical to the GPU result (the kernel's tail
/// blocks pad with zeros, which don't change the pairwise sums of real values).
fn reduce_host(a: &[f32]) -> f32 {
    let mut level: Vec<f32> = a.chunks(M).map(block_reduce).collect();
    while level.len() > 1 {
        level = level.chunks(M).map(block_reduce).collect();
    }
    level[0]
}

fn main() {
    // const N: usize = 1 << 20; // 1M elements
    const N: usize = 1 << 20;

    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();

    let a_host = uniform_vec(N, 1);
    let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();

    let module = kernels::from_module(
        ctx.load_module_from_file("vecsum.ptx")
            .expect("load vecsum.ptx"),
    )
    .expect("wrap loaded module");

    // Full recursive reduction: launch `vecsum` repeatedly until one value
    // remains (see `vecsum::reduce`).
    let gpu = reduce(&stream, &module, &a).expect("recursive reduction");

    // Two baselines, two different questions:
    //
    //   truth  (nalgebra, f64) — the accurate mathematical sum. Comparing the
    //          GPU's f32 result to this measures ACCURACY. With inputs in
    //          [-1, 1] the sum sits near zero after massive cancellation, so
    //          f32 rounding dwarfs it: this can never be a tight pass/fail gate.
    //
    //   host_ref (f32, same tree as the kernel) — reduce in M-wide blocks, then
    //          reduce those partials the same way, recursively. This tests
    //          ALGORITHM correctness: the kernel must reproduce this tree up to
    //          f32 rounding.
    let truth = DVector::from_iterator(N, a_host.iter().map(|&x| x as f64)).sum();
    let host_ref = reduce_host(&a_host);

    // The recursive kernel is deterministic (no atomics) and the host replica
    // mirrors it exactly, so in practice gpu == host_ref bit-for-bit. Keep a
    // tight guard rather than asserting exact equality, to tolerate a backend
    // that contracts adds into FMAs or reassociates: a balanced summation tree
    // over N values has depth log2(N), so its worst-case rounding error is
    // ceil(log2(N)) * eps * ||x||_1 -- far tighter than the naive N * eps bound.
    let l1: f64 = a_host.iter().map(|&x| (x as f64).abs()).sum();
    let depth = (N as f64).log2().ceil();
    let tol = depth * f32::EPSILON as f64 * l1;

    let diff = (gpu as f64 - host_ref as f64).abs();
    if diff <= tol {
        let acc_err = (gpu as f64 - truth).abs();
        println!(
            "✓ vecsum matches host replica over {N} elements (diff = {diff:.4e} ≤ tol = {tol:.4e})"
        );
        println!(
            "  accuracy vs nalgebra f64 truth = {truth:.4}: gpu = {gpu}, abs error = {acc_err:.4}"
        );
    } else {
        eprintln!(
            "✗ algorithm mismatch: gpu = {gpu}, host replica = {host_ref}, diff = {diff:.4e} > tol = {tol:.4e}"
        );
        std::process::exit(1);
    }
}
