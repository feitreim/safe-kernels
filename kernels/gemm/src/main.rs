//! GEMM (C = A · B) — correctness check.
//!
//! The kernel is defined once in `src/lib.rs` (shared with the benchmark in
//! `src/bin/bench.rs`); this binary launches it via `matmul` and checks the
//! result against an nalgebra reference product.
//!
//! Run on a GPU (via Modal):  ./run.sh gemm

use bench_util::uniform_vec;
use cuda_core::{CudaContext, DeviceBuffer};
use gemm::kernels::{self, K, M, N};
use gemm::{from_packed_bf16, matmul, to_bf16, transpose};
use nalgebra::DMatrix;

fn main() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();

    // Distinct seeds so A and B are independent draws in [-1, 1). The kernel
    // consumes bf16, so round the inputs to bf16 up front and build the
    // reference from the same rounded values — the check then measures kernel
    // correctness rather than input quantization error.
    let a_q = to_bf16(&uniform_vec(M * K, 1));
    let b_q = to_bf16(&uniform_vec(K * N, 2));
    let a = DeviceBuffer::from_host(&stream, &a_q).unwrap();
    // The kernel reads B through a K-major TMA descriptor: transpose to N×K.
    let b_t = DeviceBuffer::from_host(&stream, &transpose(&b_q, K, N)).unwrap();

    let module = kernels::from_module(
        ctx.load_module_from_file("gemm.ptx")
            .expect("load gemm.ptx"),
    )
    .expect("wrap loaded module");

    let c = matmul(&stream, &module, &a, &b_t).expect("gemm");
    let c_host = from_packed_bf16(&c.to_host_vec(&stream).unwrap());

    // Reference product on the same bf16-rounded inputs, accumulated in f32
    // like the kernel's TMEM accumulators. bf16 significands are 8 bits, so
    // each product is exact in f32; only accumulation order differs.
    let a_f: Vec<f32> = a_q.iter().map(|x| x.to_f32()).collect();
    let b_f: Vec<f32> = b_q.iter().map(|x| x.to_f32()).collect();
    let am = DMatrix::from_row_slice(M, K, &a_f);
    let bm = DMatrix::from_row_slice(K, N, &b_f);
    let truth = &am * &bm;

    let mut max_err = 0.0f64;
    let mut max_abs_t = 0.0f64;
    let (mut n_tiny, mut n_small, mut n_big) = (0usize, 0usize, 0usize);
    let mut worst: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..M {
        for j in 0..N {
            let g = c_host[i * N + j] as f64;
            let t = truth[(i, j)] as f64;
            let e = (g - t).abs();
            max_err = max_err.max(e);
            max_abs_t = max_abs_t.max(t.abs());
            // C is bf16 (8-bit significand), so at |C| ~ tens an ulp is ~0.1:
            // the fine-grained counters just profile output rounding, while
            // err > 1.0 is well past any rounding and signals a real bug.
            if e > 0.05 {
                n_tiny += 1;
            }
            if e > 0.2 {
                n_small += 1;
            }
            if e > 1.0 {
                n_big += 1;
                if worst.len() < 8 {
                    worst.push((e, i, j));
                }
            }
        }
    }
    let total = M * N;
    println!(
        "err > 0.05: {n_tiny}/{total}   err > 0.2: {n_small}/{total}   err > 1.0: {n_big}/{total}"
    );
    for (e, i, j) in &worst {
        // block coords and intra-tile coords for the 128x128 output tiles
        println!(
            "  err={e:.3} at C[{i},{j}]  block=({},{})  intra=({},{})",
            i / 128,
            j / 128,
            i % 128,
            j % 128
        );
    }

    // Two error sources: f32 accumulation-order differences, loosely bounded
    // by K²·eps as before, plus the epilogue's rounding of each C element to
    // bf16 — up to half a bf16 ulp, i.e. |C|·2⁻⁹. The bf16 term dominates.
    let acc_tol = (K as f64) * (K as f64) * f32::EPSILON as f64;
    let tol = acc_tol + max_abs_t * (2.0f64).powi(-9);

    if max_err <= tol {
        println!("✓ gemm {M}×{N}×{K}: max |err| = {max_err:.4e} ≤ tol = {tol:.4e}");
    } else {
        eprintln!("✗ gemm wrong: max |err| = {max_err:.4e} > tol = {tol:.4e}");
        std::process::exit(1);
    }
}
