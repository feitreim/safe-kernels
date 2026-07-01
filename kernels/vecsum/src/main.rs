//! Vector sum (reduction) — correctness check.
//!
//! The kernel is defined once in `src/lib.rs` (shared with the benchmark in
//! `src/bin/bench.rs`); this binary just launches it and checks the result.
//!
//! Run on a GPU (via Modal):  ./run.sh vecsum

use bench_util::uniform_vec;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use nalgebra::DVector;
use vecsum::kernels::{self, M};

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

fn main() {
    // const N: usize = 1 << 20; // 1M elements
    const N: usize = 1 << 20;

    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();

    let a_host = uniform_vec(N, 1);

    let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();

    // The host replica below reduces the input in `M`-sized blocks, so the launch
    // must use `M`-wide blocks too. `for_num_elems` picks its own default block
    // size; assert it agrees with the kernel's `M` (its `TILE` length and
    // `#[launch_bounds]`) so the two can never silently drift apart.
    let cfg = LaunchConfig::for_num_elems(N as u32);
    assert_eq!(
        cfg.block_dim.0 as usize, M,
        "launch block size {} != kernel M {M}",
        cfg.block_dim.0
    );

    let module = kernels::from_module(
        ctx.load_module_from_file("vecsum.ptx")
            .expect("load vecsum.ptx"),
    )
    .expect("wrap loaded module");
    module
        .vecsum(&stream, cfg, &a, &mut out)
        .expect("kernel launch");

    let out_host = out.to_host_vec(&stream).unwrap();
    let gpu = out_host[0];

    // Two baselines, two different questions:
    //
    //   truth  (nalgebra, f64) — the accurate mathematical sum. Comparing the
    //          GPU's f32 result to this measures ACCURACY. With inputs in
    //          [-1, 1] the sum sits near zero after massive cancellation, so
    //          f32 rounding dwarfs it: this can never be a tight pass/fail gate.
    //
    //   host_ref (f32, same order as the kernel) — M-wide interleaved pairwise
    //          per block, then accumulate the block sums. This tests ALGORITHM
    //          correctness: the kernel must reproduce it up to the reordering
    //          slop of its nondeterministic atomic accumulate.
    let truth = DVector::from_iterator(N, a_host.iter().map(|&x| x as f64)).sum();

    let block_sums: Vec<f32> = a_host.chunks(M).map(block_reduce).collect();
    let host_ref = block_sums.iter().fold(0.0f32, |acc, &b| acc + b);

    // Block sums are bit-identical on host and device; only the order in which
    // they are summed differs. Bound that reordering error by k * eps * ||b||_1.
    let l1: f64 = block_sums.iter().map(|&b| (b as f64).abs()).sum();
    let tol = block_sums.len() as f64 * f32::EPSILON as f64 * l1;

    let diff = (gpu as f64 - host_ref as f64).abs();
    if diff <= tol {
        let acc_err = (gpu as f64 - truth).abs();
        println!(
            "✓ vecsum matches host replica over {N} elements (diff = {diff:.4} ≤ tol = {tol:.4})"
        );
        println!(
            "  accuracy vs nalgebra f64 truth = {truth:.4}: gpu = {gpu}, abs error = {acc_err:.4}"
        );
    } else {
        eprintln!(
            "✗ algorithm mismatch: gpu = {gpu}, host replica = {host_ref}, diff = {diff:.4} > tol = {tol:.4}"
        );
        std::process::exit(1);
    }
}
