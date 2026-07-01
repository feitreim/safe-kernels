//! The `#[kernel]` function inside `#[cuda_module]` is compiled to PTX by the
//! cuda-oxide codegen backend and written to `vecsum.ptx` next to this crate.
//! Shared by `main.rs` (correctness check) and `src/bin/bench.rs` (throughput
//! benchmark) so the kernel is defined exactly once.
//!
//! Each binary loads the PTX file directly (`ctx.load_module_from_file` +
//! `kernels::from_module`) rather than `kernels::load`, which only looks for
//! a PTX artifact embedded in the *currently running executable*. Since the
//! kernel now lives in this lib crate rather than directly in a `main.rs`,
//! the linker drops that embedded artifact as dead weight from `main`/`bench`
//! (nothing in their compiled code references it) — `kernels::load` would
//! fail with `ModuleNotFound`. Loading from the file side-steps that.

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;
    use cuda_device::{SharedArray, atomic::DeviceAtomicF32, launch_bounds, sync_threads, warp};

    pub const M: usize = 256;
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
// Host-side recursive reduction driver.
//
// A single `vecsum` launch is only a *partial* reduction: each block collapses
// its `M`-element tile to one value, so `k` inputs become `ceil(k / M)` partial
// sums (one per block). To get a single scalar we launch again over those
// partials, and again, until one value remains — a tree of launches whose depth
// is `ceil(log_M(N))` (3 launches for N = 16M with M = 256).
//
// The kernel reads a full `M`-wide tile per block with no bounds check, so every
// buffer it consumes must be zero-padded up to a multiple of `M`. We get that
// for free: each pass's output buffer is `zeroed` at the padded size and the
// kernel writes only the live `blocks` prefix, leaving the tail zero for the
// next pass to read. Zero is the additive identity, so the padding never changes
// the sum.
use kernels::{LoadedModule, M};

fn round_up(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

/// Launch geometry for one reduction pass: `blocks` blocks of `M` threads each,
/// matching the kernel's `TILE` length and `#[launch_bounds(256)]`.
fn pass_config(blocks: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (M as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Recursively reduce `a` to a single `f32` by launching `vecsum` until one
/// value remains. `a`'s length must already be a multiple of `M` (as allocated
/// by the callers); every intermediate buffer is padded to a multiple of `M`
/// internally. Allocates a fresh output buffer per pass — fine for correctness
/// runs; the benchmark pre-allocates instead to keep allocation out of its timed
/// region.
pub fn reduce(
    stream: &CudaStream,
    module: &LoadedModule,
    a: &DeviceBuffer<f32>,
) -> Result<f32, DriverError> {
    let mut blocks = a.len().div_ceil(M);
    let mut buf = DeviceBuffer::<f32>::zeroed(stream, round_up(blocks, M))?;
    module.vecsum(stream, pass_config(blocks), a, &mut buf)?;

    while blocks > 1 {
        blocks = blocks.div_ceil(M);
        let mut next = DeviceBuffer::<f32>::zeroed(stream, round_up(blocks, M))?;
        module.vecsum(stream, pass_config(blocks), &buf, &mut next)?;
        buf = next;
    }

    Ok(buf.to_host_vec(stream)?[0])
}

// BENCHMARKS:
// -- GPU: H100                 avg time    throughput
// WARP REDUCE:                 0.0676 ms   1000.3 GB/s
// RECURSIVE:                   0.0928 ms   728.6 GB/s
// SHARED MEMORY v2:            0.1191 ms   563.6 GB/s
// SHARED MEMORY v1:            0.1328 ms   505.4 GB/s
// NAIVE (atomic add):          29.4565 ms  2.3 GB/s
