//! Shared helpers for the kernel host binaries: GPU-event timing and
//! reproducible random input generation.

use std::sync::Arc;

use cuda_core::CudaStream;

/// `n` uniform-random `f32` samples in `[-1, 1)`, from a deterministic PRNG.
///
/// Seeded so runs are reproducible: the correctness checks recompute the
/// expected result on the host, so the exact same inputs must appear on both
/// host and device. Uses splitmix64 and keeps the top 24 bits so every draw is
/// exactly representable in an `f32` mantissa.
pub fn uniform_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let unit = (z >> 40) as f32 / (1u32 << 24) as f32; // [0, 1)
            unit * 2.0 - 1.0
        })
        .collect()
}

/// Average per-iteration GPU time in milliseconds, measured with CUDA events.
///
/// Runs `warmup` untimed launches to settle clocks/caches, then times `iters`
/// launches between two recorded events (device-side timing, not wall clock).
pub fn time_gpu_iters<F>(
    stream: &Arc<CudaStream>,
    warmup: usize,
    iters: usize,
    mut launch: F,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..warmup {
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
