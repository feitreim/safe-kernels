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

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;
    use cuda_device::{SharedArray, atomic::DeviceAtomicF32, launch_bounds, sync_threads};

    pub const M: usize = 256;
    static mut TILE: SharedArray<f32, M> = SharedArray::UNINIT;

    #[kernel]
    #[launch_bounds(256)]
    pub fn vecsum(a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let i = idx.get();

        unsafe {
            TILE[tid] = a[i];
        }
        sync_threads();

        let mut s = 1;

        while s < bdim {
            let index = 2 * s * tid;

            if index < bdim {
                unsafe { TILE[index] += TILE[index + s] }
            }
            sync_threads();

            s *= 2;
        }

        if tid == 0 {
            let acc = unsafe { &*(out.as_mut_ptr() as *const DeviceAtomicF32) };
            acc.fetch_add(
                unsafe { TILE[tid] },
                cuda_device::atomic::AtomicOrdering::Relaxed,
            );
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

// BENCHMARKS:
// -- GPU: H100
// SHARED MEMORY v2:            avg=0.1191 ms   563.6 GB/s
// SHARED MEMORY v1:            avg=0.1328 ms   505.4 GB/s
// NAIVE (atomic add):          avg=29.4565 ms  2.3 GB/s
