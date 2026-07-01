//! Vector sum (reduction) — throughput microbenchmark.
//!
//! Times the kernel (defined once in `src/lib.rs`, shared with `main.rs`)
//! with CUDA events (device-side timing, not wall clock): WARMUP launches to
//! settle clocks/caches, then ITERS launches measured between two recorded
//! events. Reports average ms and effective bandwidth.
//!
//! Run on a GPU (via Modal):  ./run.sh vecsum bench

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use vecsum::kernels;

const N: usize = 1 << 24; // 16M elements
const WARMUP: usize = 5;
const ITERS: usize = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::from_module(ctx.load_module_from_file("vecsum.ptx")?)?;

    let a = DeviceBuffer::from_host(&stream, &uniform_vec(N, 1))?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let avg_ms = time_gpu_iters(&stream, WARMUP, ITERS, || {
        module.vecsum(&stream, cfg, &a, &mut out)?;
        Ok(())
    })?;

    // Memory-bound kernel: 1 read of N f32s per launch (the output write is a
    // single contended element, negligible next to the input stream).
    let gb = N as f64 * 4.0 / 1.0e9;
    let gbps = gb / (avg_ms / 1.0e3);
    println!("vecsum  N={N}  avg={avg_ms:.4} ms  {gbps:.1} GB/s");

    let out_host = out.to_host_vec(&stream)?;
    println!("✓ result (out[0] = {})", out_host[0]);
    Ok(())
}
