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

use cuda_core::{
    CudaStream, DeviceBuffer, LaunchConfig,
    sys::{
        self as cuda_sys, CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
    },
};
use cuda_device::{DisjointSlice, TmaDescriptor, cuda_module, kernel};
use std::mem::MaybeUninit;

#[cuda_module]
pub mod kernels {
    use cuda_device::{
        Barrier, SharedArray, TmaDescriptor,
        barrier::{
            fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
            mbarrier_init, mbarrier_try_wait,
        },
        sync_threads, thread,
        tma::cp_async_bulk_tensor_2d_g2s,
    };

    use super::*;

    // C[M×N] = A[M×K] · B[K×N], all row-major, all f32.
    // FLOPs = 2·M·N·K (one multiply + one add per inner-product term).
    pub const M: usize = 8192; // rows of A and C
    pub const N: usize = 1024; // cols of B and C
    pub const K: usize = 4096; // contraction dim
    pub const BK: usize = 16;
    pub const BN: usize = 16; // block is BN×BM threads
    pub const BM: usize = 16; // block is BN×BM threads
    pub const T: usize = 8; // thread computes TTILExTTILE

    // Tile dimensions
    pub const A_SIZE: usize = BM * BK * T;
    pub const B_SIZE: usize = BK * BN * T;
    pub const A_STRIDE: usize = BK;
    pub const B_STRIDE: usize = BN * T;

    #[kernel]
    pub fn gemm(
        a: *const TmaDescriptor,
        b: *const TmaDescriptor,
        mut c: DisjointSlice<f32, thread::Index2D<N>>,
    ) {
        // row-major indexing: a[row*K + k], b[k*N + col], c[row*N + col].
        // Bounds are exact multiples of TILE
        static mut TILE_A: SharedArray<f32, A_SIZE, 128> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, B_SIZE, 128> = SharedArray::UNINIT;
        const TILE_BYTES: u32 = ((A_SIZE + B_SIZE) * 4) as u32; // 4 is f32 bytes.

        static mut BAR: Barrier = Barrier::UNINIT;

        let row = thread::blockIdx_y() as usize * BM * T;
        let col = thread::blockIdx_x() as usize * BN * T;
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let tid = ty * thread::blockDim_x() as usize + tx;
        let r0 = ty * T;
        let c0 = tx * T;

        let mut acc = [[0.0f32; T]; T];
        let mut reg_a = [0.0f32; T];
        let mut reg_b = [0.0f32; T];

        // one thread initializes the barrier; the fence makes the init visible
        // to the TMA async proxy before any copy references it.
        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR, thread::blockDim_x() * thread::blockDim_y());
                fence_proxy_async_shared_cta();
            }
        }
        sync_threads();

        let mut t = 0usize;
        while t < K / BK {
            // load tile with TMA
            if tid == 0 {
                unsafe {
                    cp_async_bulk_tensor_2d_g2s(
                        &raw mut TILE_A as *mut u8,
                        a,
                        (t * BK) as i32,
                        row as i32,
                        &raw mut BAR,
                    );
                    cp_async_bulk_tensor_2d_g2s(
                        &raw mut TILE_B as *mut u8,
                        b,
                        col as i32,
                        (t * BK) as i32,
                        &raw mut BAR,
                    );
                }
            }

            // all threads arrive; the issuing thread also registers the
            // expected TMA bytes for this phase.
            let token = unsafe {
                if tid == 0 {
                    mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES)
                } else {
                    mbarrier_arrive(&raw const BAR)
                }
            };

            unsafe { while !mbarrier_try_wait(&raw const BAR, token) {} }

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
            sync_threads(); // TODO do i need this?
            t += 1;
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
                let c_ij = unsafe { c.get_unchecked_mut(c_idx) };
                *c_ij = acc[i][j];
                j += 1;
            }
            i += 1;
        }
    }

    #[kernel]
    pub fn gemm_register(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, thread::Index2D<N>>) {
        // row-major indexing: a[row*K + k], b[k*N + col], c[row*N + col].
        // Bounds are exact multiples of TILE
        static mut TILE_A: SharedArray<f32, A_SIZE> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, B_SIZE> = SharedArray::UNINIT;

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
            while fill < (BM * T * BK) / (BM * BN) {
                let a_row = fill * BM;
                let b_col = fill * BN;
                unsafe {
                    // fill rows/cols 0..T on first iter, T..T*2 on second ... etc
                    TILE_A[(ty + a_row) * A_STRIDE + tx] =
                        a[(row + ty + a_row) * K + tile_offset + tx];
                    TILE_B[ty * B_STRIDE + tx + b_col] =
                        b[(tile_offset + ty) * N + col + b_col + tx];
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

            t += 1;
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
                let c_ij = unsafe { c.get_unchecked_mut(c_idx) };
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
                TILE_A[ty * BK + tx] = a[row * K + tile_offset + tx];
                TILE_B[ty * BK + tx] = b[(tile_offset + ty) * N + col];
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

            t += 1;
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
                    sum += a[row * K + k] * b[k * N + col];
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
use kernels::{BK, BM, BN, K, LoadedModule, M, N, T};

/// Create a TMA tensor map descriptor for a 2D f32 tensor
fn create_tma_descriptor(
    global_address: *mut std::ffi::c_void,
    width: u64,
    height: u64,
    tile_width: u32,
    tile_height: u32,
) -> Result<CUtensorMap, Box<dyn std::error::Error>> {
    let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
    let tensor_rank = 2u32;
    let global_dim: [u64; 2] = [width, height];
    let global_strides: [u64; 1] = [width * std::mem::size_of::<f32>() as u64];
    let box_dim: [u32; 2] = [tile_width, tile_height];
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            tensor_rank,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != cuda_sys::cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled failed: {:?}", result).into());
    }

    Ok(unsafe { tensor_map.assume_init() })
}

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
) -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
    let mut c = DeviceBuffer::<f32>::zeroed(stream, M * N)?;
    let a_ptr = a.cu_deviceptr();
    let a_map = create_tma_descriptor(
        a_ptr as *mut std::ffi::c_void,
        K as u64,
        M as u64,
        BK as u32,
        (BM * T) as u32,
    )?;
    let dev_a_map = DeviceBuffer::from_host(stream, &a_map.opaque[..])?;
    let b_ptr = b.cu_deviceptr();
    let b_map = create_tma_descriptor(
        b_ptr as *mut std::ffi::c_void,
        N as u64,
        K as u64,
        (BN * T) as u32,
        BK as u32,
    )?;
    let dev_b_map = DeviceBuffer::from_host(stream, &b_map.opaque[..])?;
    module.gemm(
        stream,
        launch(),
        dev_a_map.cu_deviceptr() as *const TmaDescriptor,
        dev_b_map.cu_deviceptr() as *const TmaDescriptor,
        &mut c,
    )?;
    Ok(c)
}

// BASELINE
// cuBLAS SGEMM  (fill in once measured)

// BENCHMARKS:
// -- GPU (H100):               avg time        throughput
// REGISTER TILES (B=32,T=4):   avg=2.6268 ms   26161.1 GFLOPs  70.3 GB/s
// REGISTER TILES (B=8,T=8):    avg=3.9705 ms   17307.6 GFLOPs  46.5 GB/s
// SHARED+TILES:                avg=8.5085 ms   8076.5 GFLOPs   21.7 GB/s
// NAIVE:                       avg=11.1397 ms  6168.9 GFLOPs   16.6 GB/s
//
// tuning sweep (BM=BN=BK required equal by the fill pattern; T = thread tile):
//   B=8  T=8:  17.3 TFLOP/s  (64 thr)      B=16 T=8: 13.6  (256 thr, reg-bound)
//   B=16 T=4:  16.1           (256 thr)    B=32 T=4: 26.2  (1024 thr) <- best
//   B=32 T=4 + #[unroll(2)] on t-loop: 21.5 (i-cache/reg pressure, reverted)
