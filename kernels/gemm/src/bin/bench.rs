//! GEMM (C = A · B) — throughput microbenchmark.
//!
//! Times the `gemm` kernel (defined once in `src/lib.rs`, shared with `main.rs`)
//! with CUDA events (device-side timing, not wall clock): WARMUP launches to
//! settle clocks/caches, then ITERS launches measured between two recorded
//! events. GEMM is compute-bound, so the figure of merit is TFLOP/s.
//!
//! Run on a GPU (via Modal):  ./run.sh gemm bench

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer};
use gemm::kernels::{self, K, M, N};
use gemm::{Gemm, from_packed_bf16, to_bf16, transpose};

const WARMUP: usize = 500;
const ITERS: usize = 1000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::from_module(ctx.load_module_from_file("gemm.ptx")?)?;

    // bf16 inputs; B pre-transposed to N×K for the kernel's K-major descriptor.
    let a = DeviceBuffer::from_host(&stream, &to_bf16(&uniform_vec(M * K, 1)))?;
    let b_t = DeviceBuffer::from_host(&stream, &transpose(&to_bf16(&uniform_vec(K * N, 2)), K, N))?;

    // Descriptors and C are built once, outside the timed loop, so the events
    // bracket kernel launches only — no per-call allocation or H2D uploads.
    let mut gemm = Gemm::new(&stream, &a, &b_t)?;
    let avg_ms = time_gpu_iters(&stream, WARMUP, ITERS, || gemm.launch(&stream, &module))?;

    let secs = avg_ms / 1.0e3;

    // 2·M·N·K flops: one multiply + one add per contraction term.
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let tflops = flops / secs / 1.0e12;

    // Algorithmic-minimum traffic: read A and B once each, write C once. A
    // kernel with no data reuse actually pulls far more from DRAM (the naive
    // version re-reads A and B per output element), so this understates real
    // traffic — but it's the standard effective-bandwidth figure and lets
    // kernels of different tiling be compared on the same axis. A, B, and the
    // packed-bf16 C are all 2 bytes per element.
    let bytes = 2.0 * (M as f64 * K as f64 + K as f64 * N as f64 + M as f64 * N as f64);
    let gbs = bytes / secs / 1.0e9;

    println!(
        "gemm  {M}×{N}×{K}  avg={avg_ms:.4} ms  {tflops:.1} TFLOP/s  {gbs:.1} GB/s (min traffic)"
    );

    // Copy one result down so a broken launch surfaces here rather than silently.
    let c0 = from_packed_bf16(&gemm.c.to_host_vec(&stream)?[..1])[0];
    println!("✓ result (c[0] = {c0})");
    Ok(())
}
