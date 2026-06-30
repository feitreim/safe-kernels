//! Vector addition — correctness check.
//!
//! Host and device code live in this single file. The `#[kernel]` function
//! inside `#[cuda_module]` is compiled to PTX by the cuda-oxide codegen
//! backend; `main` is compiled to native code by LLVM and launches it.
//!
//! Run on a GPU (via Modal):  ./run.sh vecadd

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    }
}

fn main() {
    const N: usize = 1 << 20; // 1M elements

    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();

    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("load embedded PTX module");
    module
        .vecadd(&stream, LaunchConfig::for_num_elems(N as u32), &a, &b, &mut c)
        .expect("kernel launch");

    let c_host = c.to_host_vec(&stream).unwrap();

    let errors = (0..N).filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5).count();
    if errors == 0 {
        println!("✓ vecadd correct over {N} elements (c[..3] = {:?})", &c_host[..3]);
    } else {
        eprintln!("✗ {errors} mismatches");
        std::process::exit(1);
    }
}
