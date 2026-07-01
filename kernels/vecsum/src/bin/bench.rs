//! Vector sum (reduction) — throughput microbenchmark.
//!
//! Times the kernel (defined once in `src/lib.rs`, shared with `main.rs`)
//! with CUDA events (device-side timing, not wall clock): WARMUP launches to
//! settle clocks/caches, then ITERS launches measured between two recorded
//! events. Reports full-operation throughput (all passes) as the headline plus
//! raw pass-0 bandwidth as a diagnostic.
//!
//! Run on a GPU (via Modal):  ./run.sh vecsum bench

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use vecsum::kernels::{self, M};

const N: usize = 1 << 24; // 16M elements
const WARMUP: usize = 5;
const ITERS: usize = 50;

/// Launch geometry for one reduction pass: `blocks` blocks of `M` threads.
fn pass_config(blocks: usize) -> LaunchConfig {
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

    // `blocks[i]` is the number of live values pass i produces (one per block),
    // so it is also the block count launched for pass i: [N/M, N/M^2, ..., 1].
    let mut blocks = Vec::new();
    let mut live = N;
    while live > 1 {
        live = live.div_ceil(M);
        blocks.push(live);
    }

    // One output buffer per pass, allocated up front so malloc/free never lands
    // in the timed region. Each is padded to a multiple of M and zeroed, so the
    // next pass's tail block reads zeros (see `vecsum::reduce`).
    let mut bufs: Vec<DeviceBuffer<f32>> = blocks
        .iter()
        .map(|&b| DeviceBuffer::<f32>::zeroed(&stream, b.div_ceil(M) * M))
        .collect::<Result<_, _>>()?;

    // Full reduction: every pass. This is the number a caller of `reduce`
    // actually pays, so it drives the headline throughput.
    let avg_ms_full = time_gpu_iters(&stream, WARMUP, ITERS, || {
        // Pass 0 reads `a`; each later pass reads the previous pass's output.
        // `split_at_mut` hands out the disjoint `&input` / `&mut output` borrows.
        module.vecsum(&stream, pass_config(blocks[0]), &a, &mut bufs[0])?;
        for i in 1..bufs.len() {
            let (done, rest) = bufs.split_at_mut(i);
            module.vecsum(&stream, pass_config(blocks[i]), &done[i - 1], &mut rest[0])?;
        }
        Ok(())
    })?;

    // Pass 0 in isolation — the memory-bound launch that streams all N inputs.
    // Timed on its own so the tail passes' launch overhead and single-block
    // under-occupancy don't drag the raw kernel bandwidth down. The gap between
    // this and the full-tree throughput is exactly what the multi-pass overhead
    // costs.
    let avg_ms_pass0 = time_gpu_iters(&stream, WARMUP, ITERS, || {
        module.vecsum(&stream, pass_config(blocks[0]), &a, &mut bufs[0])?;
        Ok(())
    })?;

    // DRAM traffic for one pass of `b` blocks: each block reads an M-wide tile
    // and writes a single partial, so `b*M` reads + `b` writes.
    let pass_bytes = |b: usize| (b * M + b) as f64 * 4.0;

    // Headline: full-operation throughput. Numerator counts the bytes moved by
    // *every* pass, denominator times *every* pass — same work on both sides, so
    // the metric is internally consistent. Pass 0's N-element read is ~99% of it;
    // the tail passes add < 1% but are counted anyway to keep it honest.
    let total_gb = blocks.iter().map(|&b| pass_bytes(b)).sum::<f64>() / 1.0e9;
    let throughput = total_gb / (avg_ms_full / 1.0e3);

    // Diagnostic: raw pass-0 bandwidth — how well the streaming kernel saturates
    // HBM in isolation. This is the number to compare against the H100 peak.
    let pass0_bw = (pass_bytes(blocks[0]) / 1.0e9) / (avg_ms_pass0 / 1.0e3);

    println!(
        "vecsum  N={N}  passes={}  avg={avg_ms_full:.4} ms  throughput={throughput:.1} GB/s  [pass0 {pass0_bw:.1} GB/s]",
        bufs.len()
    );

    let result = bufs.last().unwrap().to_host_vec(&stream)?[0];
    println!("✓ result (final = {result})");
    Ok(())
}
