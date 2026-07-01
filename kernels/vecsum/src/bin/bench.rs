//! Vector sum (reduction) — throughput microbenchmark.
//!
//! Times the grid-stride `vecsum` kernel (defined once in `src/lib.rs`, shared
//! with `main.rs`) with CUDA events (device-side timing, not wall clock): WARMUP
//! launches to settle clocks/caches, then ITERS launches measured between two
//! recorded events. The reduction is a single launch, so its throughput *is* the
//! raw streaming bandwidth — directly comparable to the CUB baseline.
//!
//! Run on a GPU (via Modal):  ./run.sh vecsum bench

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use vecsum::GRID;
use vecsum::kernels::{self, M, N};

const WARMUP: usize = 5;
const ITERS: usize = 50;

/// Launch geometry: `blocks` blocks of `M` threads.
fn launch(blocks: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (M as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::from_module(ctx.load_module_from_file("vecsum.ptx")?)?;

    let a = DeviceBuffer::from_host(&stream, &uniform_vec(N, 1))?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    // One grid-stride launch: `GRID` blocks stride over all N inputs and
    // atomic-add their partials into `out`. Reusing `out` across iterations lets
    // the accumulator grow, which is harmless for timing (we redo a clean launch
    // for the correctness print below).
    let avg_ms = time_gpu_iters(&stream, WARMUP, ITERS, || {
        module.vecsum(&stream, launch(GRID), &a, &mut out)?;
        Ok(())
    })?;

    // Memory-bound: the launch reads all N inputs once; the GRID atomic adds to
    // the lone accumulator are negligible. So bandwidth is the N-element read
    // over the launch time — the same "reduce N floats" work CUB is measured on.
    let gb = N as f64 * 4.0 / 1.0e9;
    let throughput = gb / (avg_ms / 1.0e3);
    println!("vecsum  N={N}  grid={GRID}  avg={avg_ms:.4} ms  throughput={throughput:.1} GB/s");

    // Clean launch into a freshly zeroed accumulator for a correct result print
    // (the timed loop left `out` holding the sum of every iteration).
    let mut check = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.vecsum(&stream, launch(GRID), &a, &mut check)?;
    println!("✓ result (final = {})", check.to_host_vec(&stream)?[0]);
    Ok(())
}
