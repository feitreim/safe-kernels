//! GEMM (C = A · B) — throughput microbenchmark.
//!
//! Times the `gemm` kernel (defined once in `src/lib.rs`, shared with `main.rs`)
//! with CUDA events (device-side timing, not wall clock): WARMUP launches to
//! settle clocks/caches, then ITERS launches measured between two recorded
//! events. GEMM is compute-bound, so the figure of merit is GFLOP/s.
//!
//! Run on a GPU (via Modal):  ./run.sh gemm bench

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer};
use gemm::kernels::{self, K, M, N};
use gemm::matmul;

const WARMUP: usize = 5;
const ITERS: usize = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::from_module(ctx.load_module_from_file("gemm.ptx")?)?;

    let a = DeviceBuffer::from_host(&stream, &uniform_vec(M * K, 1))?;
    let b = DeviceBuffer::from_host(&stream, &uniform_vec(K * N, 2))?;

    // `matmul` allocates its own zeroed C per call; the allocation is amortized
    // over the K-deep inner products, so it doesn't distort the timing.
    let avg_ms = time_gpu_iters(&stream, WARMUP, ITERS, || {
        matmul(&stream, &module, &a, &b)?;
        Ok(())
    })?;

    let secs = avg_ms / 1.0e3;

    // 2·M·N·K flops: one multiply + one add per contraction term.
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let gflops = flops / secs / 1.0e9;

    // Algorithmic-minimum traffic: read A and B once each, write C once. A
    // kernel with no data reuse actually pulls far more from DRAM (the naive
    // version re-reads A and B per output element), so this understates real
    // traffic — but it's the standard effective-bandwidth figure and lets
    // kernels of different tiling be compared on the same axis.
    let bytes = 4.0 * (M as f64 * K as f64 + K as f64 * N as f64 + M as f64 * N as f64);
    let gbs = bytes / secs / 1.0e9;

    println!(
        "gemm  {M}×{N}×{K}  avg={avg_ms:.4} ms  {gflops:.1} GFLOP/s  {gbs:.1} GB/s (min traffic)"
    );

    // Copy one result down so a broken launch surfaces here rather than silently.
    let c = matmul(&stream, &module, &a, &b)?;
    println!("✓ result (c[0] = {})", c.to_host_vec(&stream)?[0]);
    Ok(())
}
