# cuda-oxide kernel dev environment

Write GPU kernels in pure Rust with [cuda-oxide](https://github.com/NVlabs/cuda-oxide),
build/run/benchmark them on a Modal GPU.

[cuda-oxide](https://github.com/NVlabs/cuda-oxide) is a `rustc` codegen backend
(Rust MIR → LLVM → PTX). You write host *and* device code in one Rust file:
`#[kernel]` functions become PTX, everything else compiles natively. No `.cu`
files, no `nvcc`.

The full toolchain (CUDA 13, LLVM 21, nightly `rustc-dev`, an NVIDIA GPU) only
runs on Linux+GPU, so it lives in a Modal image. You **edit kernels locally**
and **run them on Modal**.

## Layout

```
kernels/vecadd/          one standalone crate per kernel
  Cargo.toml             git deps on cuda-device/host/core (pinned tag)
  src/lib.rs             the #[cuda_module] kernel definition (shared)
  src/main.rs            correctness check  ->  ./run.sh vecadd
  src/bin/bench.rs       CUDA-event benchmark  ->  ./run.sh vecadd bench
kernels/bench-util/      shared time_gpu_iters() helper, path-dep of every
                         kernel crate's bench.rs (not run on its own)
modal_app.py             builds the toolchain image; runs kernels on a GPU
run.sh                   thin wrapper over `modal run`
```

## First-time setup

```bash
pip install modal        # if needed
modal setup              # one-time auth
```

The first run builds the Modal image (CUDA + LLVM 21 + Rust nightly + the
cuda-oxide codegen backend). That backend build is the slow part and is cached
as an image layer — later runs reuse it and only recompile your kernel.

## Use

```bash
./run.sh                 # vecadd correctness
./run.sh vecadd bench    # vecadd throughput (GB/s)
GPU=A100 ./run.sh vecadd bench
modal run modal_app.py::doctor   # GPU + toolchain sanity check
```

Default GPU is `L4`. Override with `GPU=...` (e.g. `T4`, `A10G`, `A100`, `H100`).

## Add a new kernel

Copy `kernels/vecadd` to `kernels/<name>`, set `name = "<name>"` in its
`Cargo.toml`, edit the `#[kernel]` body, then:

```bash
./run.sh <name>            # runs src/main.rs
./run.sh <name> bench      # runs src/bin/bench.rs
```

Edits are picked up on the next run — no image rebuild needed. (An image
rebuild only happens if you change `modal_app.py` or bump `CUDA_OXIDE_REF`.)

## Kernel cheatsheet

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();       // typed thread index
        let i = idx.get();                  // -> usize
        if let Some(e) = c.get_mut(idx) {   // bounds-checked output write
            *e = a[i] + b[i];
        }
    }
}
```

- `&[T]` params are read-only inputs; `DisjointSlice<T>` is a writable output
  (the `get_mut(idx)` API ties writes to the thread's own index — no aliasing).
- `kernels::load(&ctx)` loads the PTX embedded in the binary; the macro
  generates a typed `module.vecadd(&stream, cfg, &a, &b, &mut c)` launcher.
- `LaunchConfig::for_num_elems(n)` picks a 1-D grid/block for `n` threads.

## Pinned versions

`CUDA_OXIDE_REF` in `modal_app.py` and the `tag = "..."` git deps in each
`kernels/*/Cargo.toml` must match. Bump both together to upgrade.
