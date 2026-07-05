//! GEMM ladder benchmark — every kernel generation at the same M×N×K so the
//! throughputs are directly comparable.
//!
//! Run on a GPU (via Modal):  ./run.sh gemm bench_all
//!
//! `gemm_smem` is skipped: its block shape comes from the module-level BK
//! (now 64, tuned for the tcgen05 kernel), which would need a 64×64 =
//! 4096-thread block — over the 1024-thread limit. Freeze a local BK = 32
//! inside that kernel (like `gemm_register` does) to re-include it.

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::TmaDescriptor;
use gemm::kernels::{self, K, M, N};
use gemm::{Gemm, create_tma_descriptor_f32, to_bf16, transpose};

// The f32 kernels run tens to hundreds of ms per launch at 8192³, so a
// handful of iterations gives a stable average; the tcgen05 kernel is fast
// enough to want more.
const WARMUP: usize = 2;
const ITERS: usize = 10;

fn cfg(grid: (u32, u32), block: (u32, u32)) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0, grid.1, 1),
        block_dim: (block.0, block.1, 1),
        shared_mem_bytes: 0,
    }
}

fn report(name: &str, avg_ms: f64) {
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let tflops = flops / (avg_ms / 1.0e3) / 1.0e12;
    println!("{name:<14} avg={avg_ms:>9.4} ms  {tflops:>7.1} TFLOP/s");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::from_module(ctx.load_module_from_file("gemm.ptx")?)?;

    println!("gemm ladder  {M}×{N}×{K}");

    let a_host = uniform_vec(M * K, 1);
    let b_host = uniform_vec(K * N, 2);

    // f32 inputs for the pre-tcgen05 kernels; one reused f32 C.
    let a32 = DeviceBuffer::from_host(&stream, &a_host)?;
    let b32 = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut c32 = DeviceBuffer::<f32>::zeroed(&stream, M * N)?;

    // naive: one thread per C element.
    let avg = time_gpu_iters(&stream, WARMUP, ITERS, || {
        let cfg = cfg(((N / 32) as u32, (M / 32) as u32), (32, 32));
        module.gemm_naive(&stream, cfg, &a32, &b32, &mut c32)?;
        Ok(())
    })?;
    report("naive", avg);

    // register tiles: 16×16 threads × 8×8 elements/thread → 128×128 per block.
    let avg = time_gpu_iters(&stream, WARMUP, ITERS, || {
        let cfg = cfg(((N / 128) as u32, (M / 128) as u32), (16, 16));
        module.gemm_register(&stream, cfg, &a32, &b32, &mut c32)?;
        Ok(())
    })?;
    report("register", avg);

    // TMA pipeline: same 128×128 tiling as register, but the loads go through
    // f32 TMA descriptors (A: 16-wide K boxes; B: 128-wide N boxes).
    let a_map = create_tma_descriptor_f32(
        a32.cu_deviceptr() as *mut std::ffi::c_void,
        K as u64,
        M as u64,
        16,
        128,
    )?;
    let dev_a_map = DeviceBuffer::from_host(&stream, &a_map.opaque[..])?;
    let b_map = create_tma_descriptor_f32(
        b32.cu_deviceptr() as *mut std::ffi::c_void,
        N as u64,
        K as u64,
        128,
        16,
    )?;
    let dev_b_map = DeviceBuffer::from_host(&stream, &b_map.opaque[..])?;
    let avg = time_gpu_iters(&stream, WARMUP, ITERS, || {
        let cfg = cfg(((N / 128) as u32, (M / 128) as u32), (16, 16));
        module.gemm_tma_pipeline(
            &stream,
            cfg,
            dev_a_map.cu_deviceptr() as *const TmaDescriptor,
            dev_b_map.cu_deviceptr() as *const TmaDescriptor,
            &mut c32,
        )?;
        Ok(())
    })?;
    report("tma_pipeline", avg);

    // tcgen05: bf16 inputs, B pre-transposed to N×K, packed-bf16 C.
    let a16 = DeviceBuffer::from_host(&stream, &to_bf16(&a_host))?;
    let b16_t = DeviceBuffer::from_host(&stream, &transpose(&to_bf16(&b_host), K, N))?;
    let mut tcgen05 = Gemm::new(&stream, &a16, &b16_t)?;
    let avg = time_gpu_iters(&stream, 5, 50, || tcgen05.launch(&stream, &module))?;
    report("tcgen05", avg);

    println!("(gemm_smem skipped: module BK=64 would need a 4096-thread block)");
    Ok(())
}
