//! The `#[kernel]` function inside `#[cuda_module]` is compiled to PTX by the
//! cuda-oxide codegen backend and written to `gemm.ptx` next to this crate.
//! Shared by `main.rs` (correctness check) and `src/bin/bench.rs` (throughput
//! benchmark) so the kernel is defined exactly once.
//!
//! Each binary loads the PTX file directly (`ctx.load_module_from_file` +
//! `kernels::from_module`) rather than `kernels::load`, which only looks for a
//! PTX artifact embedded under this crate's name in the running executable — and
//! the linker drops that as dead weight from `main`/`bench` (nothing in their
//! compiled code references it), so `load` fails with `ModuleNotFound`. Loading
//! from the file side-steps that.

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};

#[cuda_module]
pub mod kernels {
    use cuda_device::{SharedArray, sync_threads, thread};

    use super::*;

    // C[M×N] = A[M×K] · B[K×N], all row-major, all f32.
    // FLOPs = 2·M·N·K (one multiply + one add per inner-product term).
    pub const M: usize = 8192; // rows of A and C
    pub const N: usize = 1024; // cols of B and C
    pub const K: usize = 4096; // shared / contraction dim
    pub const BK: usize = 32;
    pub const BN: usize = 32; // block is BN×BM threads
    pub const BM: usize = 32; // block is BN×BM threads
    pub const T: usize = 4; // thread computes TTILExTTILE

    // Tile dimensions
    pub const A_SIZE: usize = BM * BK * T;
    pub const B_SIZE: usize = BK * BN * T;
    pub const A_STRIDE: usize = BK;
    pub const B_STRIDE: usize = BN * T;
    static mut TILE_A: SharedArray<f32, A_SIZE> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, B_SIZE> = SharedArray::UNINIT;

    #[kernel]
    pub fn gemm(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, thread::Index2D<N>>) {
        // row-major indexing: a[row*K + k], b[k*N + col], c[row*N + col].
        // Bounds are exact multiples of TILE

        let row = thread::blockIdx_y() as usize * BM * T;
        let col = thread::blockIdx_x() as usize * BN * T;
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let r0 = ty * T;
        let c0 = tx * T;

        let mut acc = [[0.0f32; T]; T];
        let mut reg_a = [0.0f32; T];
        let mut reg_b = [0.0f32; T];

        let mut t = 0usize;
        while t < K / BK {
            let tile_offset = t * BK;

            // load tile from global to smem
            let mut fill = 0usize;
            // bound is elements/thread
            #[unroll]
            while fill < (BM*T*BK)/(BM*BN) {
                let a_row = fill * BM;
                let b_col = fill * BN;
                unsafe {
                    // fill rows/cols 0..T on first iter, T..T*2 on second ... etc
                    TILE_A[(ty + a_row) * A_STRIDE + tx] = a[(row + ty + a_row) * K + tile_offset + tx];
                    TILE_B[ty * B_STRIDE + tx + b_col] = b[(tile_offset + ty) * N + col + b_col + tx];
                }
                fill += 1;
            }
            sync_threads(); // sync after smem load

            let mut k = 0usize;
            #[unroll]
            while k < BK {
                // load from smem to registers for computation
                let mut l = 0usize;
                #[unroll]
                while l < T {
                    unsafe {
                        reg_a[l] = TILE_A[(r0 + l) * A_STRIDE + k];
                        reg_b[l] = TILE_B[k * B_STRIDE + c0 + l];
                    }
                    l += 1;
                }

                let mut i = 0usize;
                #[unroll]
                while i < T {
                    let mut j = 0usize;
                    #[unroll]
                    while j < T {
                        acc[i][j] += reg_a[i] * reg_b[j];
                        j += 1;
                    }
                    i += 1;
                }
                k += 1;
            }
            sync_threads(); // sync before wiping tiles again.

            t+=1;
        }

        // thread is responsible for TxT region of C.
        let mut i = 0usize;
        #[unroll]
        while i < T {
            let mut j = 0usize;
            #[unroll]
            while j < T {
                // have to manually calculate the correct index into C.
                let c_idx = (row + r0 + i) * N + (col + c0 + j);
                let c_ij = unsafe {c.get_unchecked_mut(c_idx)};
                *c_ij = acc[i][j];
                j += 1;
            }
            i += 1;
        }
    }

    #[kernel]
    #[allow(unused_variables)]
    pub fn gemm_smem(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, thread::Index2D<N>>) {
        // row-major indexing: a[row*K + k], b[k*N + col], c[row*N + col].
        // Bounds are exact multiples of TILE
        static mut TILE_A: SharedArray<f32, { BK * BK }> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, { BK * BK }> = SharedArray::UNINIT;

        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;

        let mut sum = 0.0f32;
        let mut t = 0usize;
        while t < K / BK {
            let tile_offset = t * BK;

            // load tile from global to smem
            unsafe {
                TILE_A[ty * BK + tx] = a[row*K + tile_offset + tx];
                TILE_B[ty * BK + tx] = b[(tile_offset + ty)*N + col];
            }
            sync_threads(); // sync after SMEM loads

            let mut k = 0usize;
            while k < BK {
                unsafe {
                    sum += TILE_A[ty * BK + k] * TILE_B[k * BK + tx];
                }
                k += 1;
            }
            sync_threads(); // sync before wiping tiles again.

            t+=1;
        }

        // const N makes this safe :)
        if let Some((c_elem, _)) = c.get_mut_indexed() {
            *c_elem += sum;
        }
    }

    #[kernel]
    #[allow(unused_variables)]
    pub fn gemm_naive(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, thread::Index2D<N>>) {
        // row-major indexing: a[row*K + k], b[k*N + col], c[row*N + col].
        // Bounds are exact multiples of TILE, so the naive version needs no
        // edge guards;
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        if let Some((c_elem, _)) = c.get_mut_indexed() {
            if row < M {
                let mut sum = 0.0f32;
                let mut k = 0;
                while k < K {
                    sum += a[row*K + k] * b[k*N + col];
                    k += 1;
                }
                *c_elem = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side GEMM driver.
//
// A single 2-D launch: a grid of (N/TILE, M/TILE) blocks of TILE×TILE threads
// covers every output element of C once. `c` is allocated per call (zeroed, so
// an unfinished kernel reads as all-zeros rather than garbage) and returned on
// the device for the caller to copy down or time.
use kernels::{LoadedModule, BM, BN, M, N, T};

/// Launch geometry: grid tiles the M×N output, one thread per C element.
/// `M` and `N` are exact multiples of `TILE`, so the grid covers C with no
/// partial edge blocks.
fn launch() -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((N / (BN * T)) as u32, (M / (BM * T)) as u32, 1),
        block_dim: (BN as u32, BM as u32, 1),
        shared_mem_bytes: 0,
    }
}

/// Compute `C = A · B` on the device, returning the M×N result buffer.
pub fn matmul(
    stream: &CudaStream,
    module: &LoadedModule,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
) -> Result<DeviceBuffer<f32>, DriverError> {
    let mut c = DeviceBuffer::<f32>::zeroed(stream, M * N)?;
    module.gemm(stream, launch(), a, b, &mut c)?;
    Ok(c)
}

// BASELINE
// cuBLAS SGEMM  (fill in once measured)

// BENCHMARKS:
// -- GPU (H100):               avg time        throughput
// REGISTER TILES (B=32,T=4):   avg=2.6268 ms   26161.1 GFLOP/s  70.3 GB/s
// REGISTER TILES (B=8,T=8):    avg=3.9705 ms   17307.6 GFLOP/s  46.5 GB/s
// SHARED+TILES:                avg=8.5085 ms   8076.5 GFLOP/s   21.7 GB/s
// NAIVE:                       avg=11.1397 ms  6168.9 GFLOP/s   16.6 GB/s
//
// tuning sweep (BM=BN=BK required equal by the fill pattern; T = thread tile):
//   B=8  T=8:  17.3 TFLOP/s  (64 thr)      B=16 T=8: 13.6  (256 thr, reg-bound)
//   B=16 T=4:  16.1           (256 thr)    B=32 T=4: 26.2  (1024 thr) <- best
//   B=32 T=4 + #[unroll(2)] on t-loop: 21.5 (i-cache/reg pressure, reverted)
