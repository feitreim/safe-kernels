//! The `#[kernel]` functions inside `#[cuda_module]` are compiled to PTX by the
//! cuda-oxide codegen backend and written to `vecsum.ptx` next to this crate.
//! Shared by `main.rs` (correctness check) and `src/bin/bench.rs` (throughput
//! benchmark) so each kernel is defined once.
//!
//! Each binary loads the PTX file directly (`ctx.load_module_from_file` +
//! `kernels::from_module`) rather than `kernels::load`, which only looks for a
//! PTX artifact embedded under this crate's name in the running executable — and
//! the linker drops that as dead weight from `main`/`bench` (nothing in their
//! compiled code references it), so `load` fails with `ModuleNotFound`. Loading
//! from the file side-steps that.

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;
    use cuda_device::{SharedArray, atomic::DeviceAtomicF32, launch_bounds, sync_threads, warp};

    pub const M: usize = 256;
    pub const N: usize = 1 << 24; // elements to reduce (16M)
    static mut TILE: SharedArray<f32, M> = SharedArray::UNINIT;

    fn warp_reduce(mut val: f32) -> f32 {
        val += warp::shuffle_xor_f32(val, 16);
        val += warp::shuffle_xor_f32(val, 8);
        val += warp::shuffle_xor_f32(val, 4);
        val += warp::shuffle_xor_f32(val, 2);
        val += warp::shuffle_xor_f32(val, 1);
        val
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn vecsum(a: &[f32], mut out: DisjointSlice<f32>) {
        //vecsum grid-stride reduction
        // we first load from global memory in chunks, thats why stride is the #
        // of total threads
        let idx = thread::index_1d();
        let tid = thread::threadIdx_x() as usize;
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;

        let mut pre = 0.0f32;
        let mut i = idx.get();
        while i < N {
            pre += a[i];
            i += stride;
        }

        unsafe {
            TILE[tid] = pre;
        }
        sync_threads();

        let mut s = thread::blockDim_x() as usize / 2;
        while s > 16 {
            if tid < s {
                unsafe { TILE[tid] += TILE[tid + s] }
            }
            sync_threads();
            s >>= 1;
        }

        // Unroll the last loop iterations once they can be handled by a single
        // warp. warps are SIMT syncronized, so they dont need additional barriers.
        let mut val = unsafe { TILE[tid] };
        if tid < 32 {
            val = warp_reduce(val);
        }

        if tid == 0 {
            let res = unsafe { &*(out.as_mut_ptr() as *const DeviceAtomicF32) };
            res.fetch_add(val, cuda_device::atomic::AtomicOrdering::Relaxed);
        }
    }


    #[kernel]
    #[launch_bounds(256)]
    pub fn vecsum_warp_unroll(a: &[f32], mut out: DisjointSlice<f32>) {
        //vecsum warp reduce
        let idx = thread::index_1d();
        let tid = thread::threadIdx_x() as usize;
        let bidx = thread::blockIdx_x() as usize;
        let i = idx.get();

        unsafe {
            TILE[tid] = a[i];
        }
        sync_threads();

        let mut s = thread::blockDim_x() as usize / 2;
        while s > 16 {
            if tid < s {
                unsafe { TILE[tid] += TILE[tid + s] }
            }
            sync_threads();
            s >>= 1;
        }

        // Unroll the last loop iterations once they can be handled by a single
        // warp. warps are SIMT syncronized, so they dont need additional barriers.
        let mut val = unsafe { TILE[tid] };
        if tid < 32 {
            val = warp_reduce(val);
        }

        if tid == 0 {
            let res = unsafe { out.get_unchecked_mut(bidx) };
            *res = val;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn vecsum_recursive(a: &[f32], mut out: DisjointSlice<f32>) {
        // sequential indexing allows for much better warp utilization, as
        // thread activity is contiguous, warp divergence is minimized, with
        // warps either fully processing a block of indices or being idle.
        // * at the end of the loop this may not be true.
        let idx = thread::index_1d();
        let tid = thread::threadIdx_x() as usize;
        let bidx = thread::blockIdx_x() as usize;
        let i = idx.get();

        unsafe {
            TILE[tid] = a[i];
        }
        sync_threads();

        // take the top half of remaining indices and add them to the bottom half
        // each iter, as s doubles, we half the number of indices.
        let mut s = thread::blockDim_x() as usize / 2;
        while s > 0 {
            if tid < s {
                unsafe { TILE[tid] += TILE[tid + s] }
            }
            sync_threads();
            s >>= 1;
        }

        if tid == 0 {
            let res = unsafe { out.get_unchecked_mut(bidx) };
            *res = unsafe { TILE[tid] };
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn vecsum_shared_v2(a: &[f32], mut out: DisjointSlice<f32>) {
        // sequential indexing allows for much better warp utilization, as
        // thread activity is contiguous, warp divergence is minimized, with
        // warps either fully processing a block of indices or being idle.
        // * at the end of the loop this may not be true.
        let idx = thread::index_1d();
        let tid = thread::threadIdx_x() as usize;
        let i = idx.get();

        unsafe {
            TILE[tid] = a[i];
        }
        sync_threads();

        // take the top half of remaining indices and add them to the bottom half
        // each iter, as s doubles, we half the number of indices.
        let mut s = thread::blockDim_x() as usize / 2;
        while s > 0 {
            if tid < s {
                unsafe { TILE[tid] += TILE[tid + s] }
            }
            sync_threads();
            s >>= 1;
        }

        if tid == 0 {
            let res = unsafe { &*(out.as_mut_ptr() as *const DeviceAtomicF32) };
            res.fetch_add(unsafe { TILE[tid] }, cuda_device::atomic::AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub fn vecsum_shared_v1(a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let tid = thread::threadIdx_x() as usize;
        let i = idx.get();

        unsafe {
            TILE[tid] = a[i];
        }
        sync_threads();

        if tid == 0 {
            let mut sum = 0.0f32;
            let mut j: usize = 0;
            while j < thread::blockDim_x() as usize {
                sum += unsafe { TILE[j] };
                j += 1;
            }

            let acc = unsafe { &*(out.as_mut_ptr() as *const DeviceAtomicF32) };
            acc.fetch_add(sum, cuda_device::atomic::AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub fn vecsum_naive(a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let acc = unsafe { &*(out.as_mut_ptr() as *const DeviceAtomicF32) };
        acc.fetch_add(a[i], cuda_device::atomic::AtomicOrdering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Host-side reduction driver.
//
// `vecsum` is a single-launch grid-stride reduction: a fixed grid of `GRID`
// blocks strides over all `N` inputs (a compile-time const in the kernel), each
// block reduces its share to one value, and each block's thread 0 atomic-adds
// that partial into the lone output accumulator. One launch — no recursive tree,
// and since the kernel bounds its reads with `i < N` there is nothing to
// zero-pad. `out` must start at zero because every block accumulates into it;
// the atomic adds land in unspecified order, so the result is not bit-
// reproducible run to run.
use kernels::{LoadedModule, M};

/// Grid width: how many blocks stride over the input. Sized to roughly fill an
/// H100 — ~full occupancy at 256 threads/block — and the main knob to sweep for
/// the streaming pass.
pub const GRID: usize = 1024;

/// Launch geometry: `blocks` blocks of `M` threads, matching the kernel's `TILE`
/// length and `#[launch_bounds(256)]`.
fn launch(blocks: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (M as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Reduce the `kernels::N` elements of `a` to a single `f32` in one grid-stride
/// launch. `out` starts zeroed because the kernel accumulates into it with
/// atomic adds.
pub fn reduce(
    stream: &CudaStream,
    module: &LoadedModule,
    a: &DeviceBuffer<f32>,
) -> Result<f32, DriverError> {
    let mut out = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    module.vecsum(stream, launch(GRID), a, &mut out)?;
    Ok(out.to_host_vec(stream)?[0])
}

// BASELINE
// cub::DeviceReduce  avg=0.0274 ms  throughput=2452.9 GB/s

// BENCHMARKS:
// -- GPU: H100                 avg time    throughput
// GRID-STRIDE:                 0.0334 ms   2008.8 GB/s
// WARP UNROLL:                 0.0676 ms   1000.3 GB/s
// RECURSIVE:                   0.0928 ms   728.6 GB/s
// SHARED MEMORY v2:            0.1191 ms   563.6 GB/s
// SHARED MEMORY v1:            0.1328 ms   505.4 GB/s
// NAIVE (atomic add):          29.4565 ms  2.3 GB/s
