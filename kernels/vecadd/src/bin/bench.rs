//! Vector addition — throughput microbenchmark.
//!
//! Times the kernel with CUDA events (device-side timing, not wall clock):
//! WARMUP launches to settle clocks/caches, then ITERS launches measured
//! between two recorded events. Reports average ms and effective bandwidth.
//!
//! Run on a GPU (via Modal):  ./run.sh vecadd bench

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const N: usize = 1 << 24; // 16M elements
const WARMUP: usize = 5;
const ITERS: usize = 50;

#[cuda_module]
mod kernels {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let a = DeviceBuffer::from_host(&stream, &vec![1.0f32; N])?;
    let b = DeviceBuffer::from_host(&stream, &vec![2.0f32; N])?;
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let avg_ms = time_gpu_iters(&stream, ITERS, || {
        module.vecadd(&stream, cfg, &a, &b, &mut c)?;
        Ok(())
    })?;

    // Memory-bound kernel: 2 reads + 1 write of N f32s per launch.
    let gb = 3.0 * N as f64 * 4.0 / 1.0e9;
    let gbps = gb / (avg_ms / 1.0e3);
    println!("vecadd  N={N}  avg={avg_ms:.4} ms  {gbps:.1} GB/s");

    let c_host = c.to_host_vec(&stream)?;
    assert!((c_host[0] - 3.0).abs() < 1e-5, "wrong result: {}", c_host[0]);
    println!("✓ result verified (c[0] = {})", c_host[0]);
    Ok(())
}

/// Average per-iteration GPU time in milliseconds, measured with CUDA events.
fn time_gpu_iters<F>(
    stream: &Arc<CudaStream>,
    iters: usize,
    mut launch: F,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..WARMUP {
        launch()?;
    }
    stream.synchronize()?;

    let flags = cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
    let start = stream.record_event(Some(flags))?;
    for _ in 0..iters {
        launch()?;
    }
    let end = stream.record_event(Some(flags))?;
    Ok(start.elapsed_ms(&end)? as f64 / iters as f64)
}
