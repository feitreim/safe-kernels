//! Vector addition — throughput microbenchmark.
//!
//! Times the kernel (defined once in `src/lib.rs`, shared with `main.rs`)
//! with CUDA events (device-side timing, not wall clock): WARMUP launches to
//! settle clocks/caches, then ITERS launches measured between two recorded
//! events. Reports average ms and effective bandwidth.
//!
//! Run on a GPU (via Modal):  ./run.sh vecadd bench

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use vecadd::kernels;

const N: usize = 1 << 24; // 16M elements
const WARMUP: usize = 5;
const ITERS: usize = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::from_module(ctx.load_module_from_file("vecadd.ptx")?)?;

    let a_host = uniform_vec(N, 1);
    let b_host = uniform_vec(N, 2);
    let a = DeviceBuffer::from_host(&stream, &a_host)?;
    let b = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let avg_ms = time_gpu_iters(&stream, WARMUP, ITERS, || {
        module.vecadd(&stream, cfg, &a, &b, &mut c)?;
        Ok(())
    })?;

    // Memory-bound kernel: 2 reads + 1 write of N f32s per launch.
    let gb = 3.0 * N as f64 * 4.0 / 1.0e9;
    let gbps = gb / (avg_ms / 1.0e3);
    println!("vecadd  N={N}  avg={avg_ms:.4} ms  {gbps:.1} GB/s");

    let c_host = c.to_host_vec(&stream)?;
    let expected = a_host[0] + b_host[0];
    assert!(
        (c_host[0] - expected).abs() < 1e-5,
        "wrong result: {} != {expected}",
        c_host[0]
    );
    println!("✓ result verified (c[0] = {})", c_host[0]);
    Ok(())
}
