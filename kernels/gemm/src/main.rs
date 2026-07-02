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
use gemm::matmul;
use nalgebra::DMatrix;

fn main() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();

    // Distinct seeds so A and B are independent draws in [-1, 1).
    let a_host = uniform_vec(M * K, 1);
    let b_host = uniform_vec(K * N, 2);
    let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b = DeviceBuffer::from_host(&stream, &b_host).unwrap();

    let module = kernels::from_module(
        ctx.load_module_from_file("gemm.ptx")
            .expect("load gemm.ptx"),
    )
    .expect("wrap loaded module");

    let c = matmul(&stream, &module, &a, &b).expect("gemm");
    let c_host = c.to_host_vec(&stream).unwrap();

    // Reference product. Our buffers are row-major; `from_row_slice` reads them
    // in that layout, and DMatrix indexing `[(i, j)]` is layout-agnostic, so we
    // can compare element-wise against the row-major GPU buffer directly.
    let am = DMatrix::from_row_slice(M, K, &a_host);
    let bm = DMatrix::from_row_slice(K, N, &b_host);
    let truth = &am * &bm;

    // Each C element is an f32 sum of K products, each product in [-1, 1). The
    // running-sum rounding error is bounded by ~(K-1)·eps·Σ|terms| ≤ K²·eps, so
    // that is a safe (loose) absolute tolerance — well above the ~K·eps you
    // actually see on random inputs, but tight enough to catch any real bug.
    let tol = (K as f64) * (K as f64) * f32::EPSILON as f64;

    let mut max_err = 0.0f64;
    for i in 0..M {
        for j in 0..N {
            let g = c_host[i * N + j] as f64;
            let t = truth[(i, j)] as f64;
            max_err = max_err.max((g - t).abs());
        }
    }

    if max_err <= tol {
        println!("✓ gemm {M}×{N}×{K}: max |err| = {max_err:.4e} ≤ tol = {tol:.4e}");
    } else {
        eprintln!("✗ gemm wrong: max |err| = {max_err:.4e} > tol = {tol:.4e}");
        std::process::exit(1);
    }
}
