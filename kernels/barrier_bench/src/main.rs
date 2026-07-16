//! Stage-selection benchmark: the real gemm kernel in three selection idioms
//! (see `src/lib.rs`): `match` yielding selected values, `match` with the
//! body duplicated per arm, and stage-indexed arrays.
//!
//! All variants run on the same inputs and descriptors. Per-tile math and
//! accumulation order are identical, so their C outputs must match bitwise —
//! checked once before timing. GEMM is compute-bound, so the figure of merit
//! is TFLOP/s; the selection idiom only moves the number through the mbarrier
//! handshake overhead it adds to every pipeline stage.
//!
//! Run on a GPU (via Modal):  ./run.sh barrier_bench

use barrier_bench::kernels::{self, K, M, N};
use barrier_bench::{Gemm, from_packed_bf16, to_bf16, transpose};
use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer};

const WARMUP: usize = 500;
const ITERS: usize = 1000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::from_module(ctx.load_module_from_file("barrier_bench.ptx")?)?;

    // bf16 inputs; B pre-transposed to N×K for the kernel's K-major descriptor.
    let a = DeviceBuffer::from_host(&stream, &to_bf16(&uniform_vec(M * K, 1)))?;
    let b_t = DeviceBuffer::from_host(&stream, &transpose(&to_bf16(&uniform_vec(K * N, 2)), K, N))?;
    let mut gemm = Gemm::new(&stream, &a, &b_t)?;

    // Cross-check first: same inputs + same accumulation order, so all
    // variants must agree bitwise. A wrong barrier selection shows up here as
    // corrupted tiles (or a hang) rather than a silently bogus timing.
    gemm.launch_arith(&stream, &module)?;
    let c_arith = gemm.c.to_host_vec(&stream)?;
    let mismatches = |c: &[u32]| c.iter().zip(&c_arith).filter(|(v, a)| v != a).count();

    gemm.launch_match(&stream, &module)?;
    let match_bad = mismatches(&gemm.c.to_host_vec(&stream)?);
    gemm.launch_branch(&stream, &module)?;
    let branch_bad = mismatches(&gemm.c.to_host_vec(&stream)?);

    let match_ms = time_gpu_iters(&stream, WARMUP, ITERS, || gemm.launch_match(&stream, &module))?;
    let branch_ms =
        time_gpu_iters(&stream, WARMUP, ITERS, || gemm.launch_branch(&stream, &module))?;
    let arith_ms = time_gpu_iters(&stream, WARMUP, ITERS, || gemm.launch_arith(&stream, &module))?;

    // 2·M·N·K flops: one multiply + one add per contraction term.
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let tflops = |ms: f64| flops / (ms / 1.0e3) / 1.0e12;

    println!("gemm stage selection  ({M}×{N}×{K}, 4 stages, 2 accum stages)");
    for (name, ms) in [
        ("match over statics ", match_ms),
        ("branch per stage   ", branch_ms),
        ("stage-indexed arrays", arith_ms),
    ] {
        println!(
            "{name}  avg={ms:.4} ms  {:.1} TFLOP/s  ({:.2}x vs arrays)",
            tflops(ms),
            ms / arith_ms
        );
    }

    if match_bad == 0 && branch_bad == 0 {
        let c0 = from_packed_bf16(&c_arith[..1])[0];
        println!("✓ outputs bitwise identical (c[0] = {c0})");
        Ok(())
    } else {
        println!(
            "✗ outputs differ vs arith: match {match_bad}, branch {branch_bad} of {} u32s",
            c_arith.len()
        );
        std::process::exit(1);
    }
}
