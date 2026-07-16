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
        self as cuda_sys, CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
    },
};
use cuda_device::{DisjointSlice, TmaDescriptor, cuda_module, kernel};
use std::mem::MaybeUninit;

#[cuda_module]
pub mod kernels {
    use cuda_device::{
        Barrier, ManagedBarrier, MmaBarrier, SharedArray, TmaBarrier, TmaDescriptor, TmemUninit,
        Uninit,
        barrier::{
            fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
            mbarrier_init, mbarrier_inval, mbarrier_try_wait,
        },
        clc::{clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel},
        cluster_launch, sync_threads,
        tcgen05::{
            Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor,
            Tcgen05MmaShape, Tcgen05SmemDescriptor, Tcgen05SwizzleMode, TmemGuard,
            cvt_f32x2_bf16x2, stmatrix_m8n8_x2, tcgen05_commit_shared_cluster,
            tcgen05_ld_16x256b_pure, tcgen05_load_wait, tcgen05_mma_f16,
        },
        thread,
        tma::cp_async_bulk_tensor_2d_g2s,
        warp,
    };

    use super::*;

    // C[M×N] = A[M×K] · B[K×N], all row-major, all f32.
    // FLOPs = 2·M·N·K (one multiply + one add per inner-product term).
    pub const M: usize = 8192; // rows of A and C
    pub const N: usize = 8192; // cols of B and C
    pub const K: usize = 8192; // contraction dim
    pub const BM: usize = 128;
    pub const BN: usize = 128;
    pub const BK: usize = 64;

    // CLC tile scheduling (work stealing) + TMEM accumulator pipeline.
    // warp 4 exclusively handles TMA
    // warp 5 exclusively handles MMA
    // the GPU creates a total_tiles length queue with blocks of 4 CTAs (cluster dim)
    // CTA flow:
    // 1. Process own tile
    //
    // 2. steal a cluster
    // lane 0: arm + issue `clc_try_cancel`
    // then wait/decode answer, then shuffle the result across the warp.
    //
    // 3. process the 4 stolen tiles serially
    //
    // 4. repeat until canceled.
    //
    // tile m,n are linear assigned COLUMN major from C.
    // tile_m = blockIdx_x % TILES_M
    // tile_n = blockIdx_x / TILES_M
    // This means that consecutive cta_IDs are adjacent in (the row-major) memory.
    //
    // Launch dims:
    // grid dim     = (total_tiles, 1, 1)
    // cluster dim  = (4, 1, 1)
    // block dim    = (192, 1, 1)
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub fn gemm(
        a: *const TmaDescriptor,
        b: *const TmaDescriptor, // reminder, B should be transposed to be NxK first
        mut c: DisjointSlice<u32>,
    ) {
        // SMEM tiles contain f16/bf16 and are shape:
        // A: BM x BK
        // B: BK x BN
        // MMA instructions process MMA_K=16 at a time, so we have BK/MMA_K (4) MMA instructions
        const BF16_SIZE: usize = 2;
        const A_TILE_BYTES: usize = BM * BK * BF16_SIZE; // * 2 bc f16 is 2x u8
        const B_TILE_BYTES: usize = BK * BN * BF16_SIZE;
        const COMBINED_BYTES: u32 = (A_TILE_BYTES + B_TILE_BYTES) as u32;
        const OUTPUT_SIZE: usize = BM * BN / BF16_SIZE;
        const CLUSTER_SIZE: u32 = 4;
        const TILES_M: u32 = (M / BM) as u32;
        const STAGES: u32 = 4;
        const NUM_ACCUM_STAGES: u32 = 2;

        // setup smem
        static mut SMEM_A0: SharedArray<u8, A_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut SMEM_A1: SharedArray<u8, A_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut SMEM_A2: SharedArray<u8, A_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut SMEM_A3: SharedArray<u8, A_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut SMEM_B0: SharedArray<u8, B_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut SMEM_B1: SharedArray<u8, B_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut SMEM_B2: SharedArray<u8, B_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut SMEM_B3: SharedArray<u8, B_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut SMEM_OUT: SharedArray<u32, OUTPUT_SIZE, 128> = SharedArray::UNINIT;
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
        // We'll also need an output SMEM now, BM x BN, these are bf16 packed into u32

        // TMA/MMA double buffered
        static mut TMA_BARS: [Barrier; 4] = [Barrier::UNINIT,Barrier::UNINIT,Barrier::UNINIT,Barrier::UNINIT];
        static mut MMA_BARS: [Barrier; 4] = [Barrier::UNINIT,Barrier::UNINIT,Barrier::UNINIT,Barrier::UNINIT];

        // TMEM accumulator pipelining
        static mut ACCUM_FULL_0: Barrier = Barrier::UNINIT;
        static mut ACCUM_FULL_1: Barrier = Barrier::UNINIT;
        static mut ACCUM_EMPTY_0: Barrier = Barrier::UNINIT;
        static mut ACCUM_EMPTY_1: Barrier = Barrier::UNINIT;
        static mut TILE_READY: Barrier = Barrier::UNINIT;

        // CLC (workstealing) response buffer and barrier
        static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;
        static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
        static mut CLC_BAR: Barrier = Barrier::UNINIT;
        // tile info layout:
        // +0 : tile_m
        // +1 : tile_n
        // +2 : 0 = finished, 1 = working

        const SBO_BYTES: u32 = 1024;
        const LBO_BYTES: u32 = 16;
        const TMA_WARP: u32 = 4;
        const MMA_WARP: u32 = 5;

        // live info:
        let tid = thread::threadIdx_x();
        let warp_id = warp::warp_id();
        let lane_id = tid % 32;
        let thread0 = tid == 0;

        // --- Stage 0: Initialize Barriers and TMEM ---
        let tma_bars = unsafe {
            TMA_BARS.map(|bar| ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut bar))
        };
        let mma_bars = unsafe {
            MMA_BARS.map(|bar| ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut bar))
        };
        let accum_full_0 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_FULL_0);
        let accum_full_1 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_FULL_1);
        let accum_empty_0 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_EMPTY_0);
        let accum_empty_1 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_EMPTY_1);
        let tile_bar = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut TILE_READY);
        let clc_bar = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut CLC_BAR);

        let tma_bars = tma_bars.map(|bar| unsafe {bar.init(1)});
        let mma_bars = mma_bars.map(|bar| unsafe {bar.init(1)});

        let accum_full_0 = unsafe { accum_full_0.init(1) };
        let accum_full_1 = unsafe { accum_full_1.init(1) };
        let accum_empty_0 = unsafe { accum_empty_0.init(128) }; // number of epilogue threads
        let accum_empty_1 = unsafe { accum_empty_1.init(128) };
        let tile_bar = unsafe { tile_bar.init(1) };
        let clc_bar = unsafe { clc_bar.init(1) };

        let tmem = TmemGuard::<TmemUninit, { BN as u32 * NUM_ACCUM_STAGES}>::from_static(
            &raw mut TMEM_ADDR as *mut u32,
        );
        let tmem = unsafe { tmem.alloc() };
        thread::sync_threads();

        let mma_desc = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M128_N128)
            .element_type(Tcgen05ElementType::BF16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();

        // pre signal mma bars so they don't stall on the first iter:
        if thread0 {
            mma_bar_0.arrive();
            mma_bar_1.arrive();
            mma_bar_2.arrive();
            mma_bar_3.arrive();
        }
        sync_threads();

        // --- Stage 1: K-Loop ---
        // Warp Specialized Version
        // Warp 4: TMA Producer loads tiles
        // Warp 5: MMA Consumer computes tiles
        // Warp 0-3: Idle
        let k_iters = (K / BK) as u32;

        if warp_id == TMA_WARP {
            let is_lane0 = warp::lane_id() == 0;
            let mut global_k: u32 = 0;
            let mut clc_iter: u32 = 0;

            // consume the starter tile first:
            let first_ctaid = thread::blockIdx_x();
            let first_tile_m = first_ctaid % TILES_M;
            let first_tile_n = first_ctaid / TILES_M;

            if is_lane0 {
                unsafe {
                    *(&raw mut TILE_INFO as *mut u32).add(0) = first_tile_m;
                    *(&raw mut TILE_INFO as *mut u32).add(1) = first_tile_n;
                    *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                }
                tile_bar.arrive();
            }

            let m_offset = (first_tile_m * BM as u32) as i32;
            let n_offset = (first_tile_n * BN as u32) as i32;

            let mut k_idx = 0u32;
            while k_idx < k_iters {
                let phase = global_k % STAGES;
                let parity = (global_k / STAGES) & 1;

                let (smem_a, smem_b, mma_bar, tma_bar) = match phase {
                    0 => (
                        &raw mut SMEM_A0 as *mut u8,
                        &raw mut SMEM_B0 as *mut u8,
                        &mma_bar_0,
                        &tma_bar_0,
                    ),
                    1 => (
                        &raw mut SMEM_A1 as *mut u8,
                        &raw mut SMEM_B1 as *mut u8,
                        &mma_bar_1,
                        &tma_bar_1,
                    ),
                    2 => (
                        &raw mut SMEM_A2 as *mut u8,
                        &raw mut SMEM_B2 as *mut u8,
                        &mma_bar_2,
                        &tma_bar_2,
                    ),
                    _ => (
                        &raw mut SMEM_A3 as *mut u8,
                        &raw mut SMEM_B3 as *mut u8,
                        &mma_bar_3,
                        &tma_bar_3,
                    ),
                };

                // Wait for MMA computation to finish, freeing the tile.
                while !mma_bar.try_wait_parity(parity) {}

                if is_lane0 {
                    let k_offset = (k_idx as usize * BK) as i32;
                    unsafe {
                        cp_async_bulk_tensor_2d_g2s(
                            smem_a,
                            a,
                            k_offset,
                            m_offset,
                            tma_bar.as_ptr() as *mut Barrier,
                        );
                        cp_async_bulk_tensor_2d_g2s(
                            smem_b,
                            b,
                            k_offset,
                            n_offset,
                            tma_bar.as_ptr() as *mut Barrier,
                        );
                    }
                    tma_bar.arrive_expect_tx(COMBINED_BYTES);
                }
                k_idx += 1;
                global_k += 1;
            }

            let clc_ptr = &raw mut CLC_RESPONSE as *mut u8;

            loop {
                let clc_parity = clc_iter & 1;

                // keep response logic in lane 0.
                let mut is_canceled = 0u32;
                let mut first_stolen = 0u32;
                if is_lane0 {
                    unsafe {
                        fence_proxy_async_shared_cta();
                        clc_bar.arrive_expect_tx(16); // CLC response size is 16 bytes
                        clc_try_cancel(clc_ptr, &raw mut CLC_BAR);

                        // wait on clc barrier
                        while !clc_bar.try_wait_parity(clc_parity) {}
                        let resp_lo = CLC_RESPONSE[0];
                        let resp_hi = CLC_RESPONSE[1];

                        is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
                        if is_canceled != 0 {
                            first_stolen = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
                        }
                        fence_proxy_async_shared_cta();
                    }
                }

                is_canceled = warp::shuffle(is_canceled, 0);
                first_stolen = warp::shuffle(first_stolen, 0);

                // bail if canceled
                if is_canceled == 0 {
                    if is_lane0 {
                        // update TILE_INFO with cancellation
                        unsafe {
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 0;
                        }
                        tile_bar.arrive();
                    }
                    break;
                }

                let mut cluster_step = 0u32;
                while cluster_step < CLUSTER_SIZE {
                    let ctaid = first_stolen + cluster_step;
                    let tile_m = ctaid % TILES_M;
                    let tile_n = ctaid / TILES_M;

                    if is_lane0 {
                        unsafe {
                            *(&raw mut TILE_INFO as *mut u32).add(0) = tile_m;
                            *(&raw mut TILE_INFO as *mut u32).add(1) = tile_n;
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                        }
                        tile_bar.arrive();
                    }

                    let m_offset = (tile_m * BM as u32) as i32;
                    let n_offset = (tile_n * BN as u32) as i32;

                    let mut k_idx = 0u32;
                    while k_idx < k_iters {
                        let phase = global_k % STAGES;
                        let parity = (global_k / STAGES) & 1;

                        let (smem_a, smem_b, mma_bar, tma_bar) = match phase {
                            0 => (
                                &raw mut SMEM_A0 as *mut u8,
                                &raw mut SMEM_B0 as *mut u8,
                                &mma_bar_0,
                                &tma_bar_0,
                            ),
                            1 => (
                                &raw mut SMEM_A1 as *mut u8,
                                &raw mut SMEM_B1 as *mut u8,
                                &mma_bar_1,
                                &tma_bar_1,
                            ),
                            2 => (
                                &raw mut SMEM_A2 as *mut u8,
                                &raw mut SMEM_B2 as *mut u8,
                                &mma_bar_2,
                                &tma_bar_2,
                            ),
                            _ => (
                                &raw mut SMEM_A3 as *mut u8,
                                &raw mut SMEM_B3 as *mut u8,
                                &mma_bar_3,
                                &tma_bar_3,
                            ),
                        };

                        // Wait for MMA computation to finish, freeing the tile.
                        while !mma_bar.try_wait_parity(parity) {}

                        if is_lane0 {
                            let k_offset = (k_idx as usize * BK) as i32;
                            unsafe {
                                cp_async_bulk_tensor_2d_g2s(
                                    smem_a,
                                    a,
                                    k_offset,
                                    m_offset,
                                    tma_bar.as_ptr() as *mut Barrier,
                                );
                                cp_async_bulk_tensor_2d_g2s(
                                    smem_b,
                                    b,
                                    k_offset,
                                    n_offset,
                                    tma_bar.as_ptr() as *mut Barrier,
                                );
                            }
                            tma_bar.arrive_expect_tx(COMBINED_BYTES);
                        }
                        k_idx += 1;
                        global_k += 1;
                    }
                    cluster_step += 1;
                }
                clc_iter += 1;
            }
        }

        if warp_id == MMA_WARP {
            let is_lane0 = lane_id == 0;
            let mut tile_iter = 0u32;
            let mut tile_parity = 0u32;
            let mut global_k = 0u32;

            loop {
                while !tile_bar.try_wait_parity(tile_parity) {}
                tile_parity ^= 1;

                let has_work = unsafe { *(&raw const TILE_INFO as *const u32).add(2) };
                if has_work == 0 {
                    break;
                }

                let accum_stage = tile_iter % NUM_ACCUM_STAGES;
                let tmem_stage_offset = accum_stage * BN as u32;
                let (accum_empty, accum_full) = match accum_stage {
                    0 => (&accum_empty_0, &accum_full_0),
                    _ => (&accum_empty_1, &accum_full_1),
                };

                if tile_iter >= NUM_ACCUM_STAGES {
                    let empty_parity = ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                    while !accum_empty.try_wait_parity(empty_parity) {}
                }

                let mut k_idx = 0u32;
                while k_idx < k_iters {
                    let phase = global_k % STAGES;
                    let parity = (global_k / STAGES) & 1;

                    let (smem_a, smem_b, mma_bar, tma_bar) = match phase {
                        0 => (
                            &raw mut SMEM_A0 as u64,
                            &raw mut SMEM_B0 as u64,
                            &mma_bar_0,
                            &tma_bar_0,
                        ),
                        1 => (
                            &raw mut SMEM_A1 as u64,
                            &raw mut SMEM_B1 as u64,
                            &mma_bar_1,
                            &tma_bar_1,
                        ),
                        2 => (
                            &raw mut SMEM_A2 as u64,
                            &raw mut SMEM_B2 as u64,
                            &mma_bar_2,
                            &tma_bar_2,
                        ),
                        _ => (
                            &raw mut SMEM_A3 as u64,
                            &raw mut SMEM_B3 as u64,
                            &mma_bar_3,
                            &tma_bar_3,
                        ),
                    };

                    // wait for tile memory to be filled
                    while !tma_bar.try_wait_parity(parity) {}

                    if is_lane0 {
                        let mut k_sub = 0;
                        while k_sub < BK as u64 / 16 {
                            let k_offset = (k_sub * 32) as u64;
                            let a_desc = Tcgen05SmemDescriptor::from_bytes(
                                smem_a + k_offset,
                                LBO_BYTES,
                                SBO_BYTES,
                                Tcgen05SwizzleMode::Swizzle128B,
                            );
                            let b_desc = Tcgen05SmemDescriptor::from_bytes(
                                smem_b + k_offset,
                                LBO_BYTES,
                                SBO_BYTES,
                                Tcgen05SwizzleMode::Swizzle128B,
                            );

                            // accumulate into d always except for the very first time.
                            let accumlate = k_idx > 0 || k_sub > 0;
                            unsafe {
                                tcgen05_mma_f16(
                                    tmem.address().raw() + tmem_stage_offset,
                                    a_desc.raw(),
                                    b_desc.raw(),
                                    mma_desc.raw(),
                                    accumlate,
                                )
                            };
                            k_sub += 1;
                        }

                        unsafe {
                            tcgen05_commit_shared_cluster(mma_bar.as_ptr() as *mut u64);
                        }
                    }
                    k_idx += 1;
                    global_k += 1;
                }

                if is_lane0 {
                    unsafe {
                        tcgen05_commit_shared_cluster(accum_full.as_ptr() as *mut u64);
                    }
                }

                tile_iter += 1;
            }
        }

        // --- Stage 2: TMEM -> Registers -> SMEM ---
        // Now two phases, for each accumulator.
        if warp_id < 4 {
            let mut epilogue_tile_iter = 0u32;
            let mut tile_parity = 0u32;
            // TMEM is BM x BN (128 x 128)
            // values are f32
            // tcgen05_ld_16x256b_pure loads 4x f32 per thread
            // loads 16 rows x 8 columns
            // so we load 2x matrices with a col offset of 16
            // when we pass the address its packed (row << 16) | column
            // so the addr is tmem.address + (tmem_row << 16) | col_offset
            // lines up exactly with stmatrix_m8n8_x2
            // The load command we are using (stmatrix_m8n8_x2) takes 2 8x8
            // matrices, it is designed to match the output from the tmem loads we
            // used. with that in mind, lanes 0-7 compute the row addresses for the
            // first matrix, and lanes 8-15 compute the row addresses for the
            // second.
            let warp_row = (warp_id * 32) as usize;
            let row_stride_bytes = BN * 2;
            let col_step = 16;
            let second_load_offset = 8; // high columns
            let row_within_8 = (lane_id % 8) as usize;
            let is_second_matrix = (8..16).contains(&lane_id);
            let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

            loop {
                while !tile_bar.try_wait_parity(tile_parity) {}
                tile_parity ^= 1;

                let info_ptr = &raw const TILE_INFO as *const u32;
                let has_work = unsafe { *info_ptr.add(2) };
                if has_work == 0 {
                    break;
                }

                // Tile to clear!
                let tile_m = unsafe { *info_ptr.add(0) };
                let tile_n = unsafe { *info_ptr.add(1) };

                let accum_stage = epilogue_tile_iter % NUM_ACCUM_STAGES;

                let (full_bar, empty_bar, tmem_stage_offset) = match accum_stage {
                    0 => (&accum_full_0, &accum_empty_0, 0u32),
                    _ => (&accum_full_1, &accum_empty_1, BN as u32),
                };

                let full_parity = (epilogue_tile_iter / NUM_ACCUM_STAGES) & 1;

                while !full_bar.try_wait_parity(full_parity) {}

                let mut tmem_row_offset = 0u32;
                #[unroll]
                while tmem_row_offset < 32 {
                    let tmem_row = warp_row as u32 + tmem_row_offset;

                    let mut col_block = 0u32;
                    while col_block < 8 {
                        let col_offset = (col_block * col_step) as usize;
                        unsafe {
                            let regs_a = tcgen05_ld_16x256b_pure(
                                tmem.address().raw()
                                + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32,
                            );
                            tcgen05_load_wait();

                            let regs_b = tcgen05_ld_16x256b_pure(
                                tmem.address().raw()
                                + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32
                                    + second_load_offset,
                            );
                            tcgen05_load_wait();

                            let p0_lo = cvt_f32x2_bf16x2(regs_a.x(), regs_a.y());
                            let p1_lo = cvt_f32x2_bf16x2(regs_b.x(), regs_b.y());

                            let out_row_lo = tmem_row as usize + row_within_8;
                            let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_lo * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                            let p0_hi = cvt_f32x2_bf16x2(regs_a.z(), regs_a.w());
                            let p1_hi = cvt_f32x2_bf16x2(regs_b.z(), regs_b.w());
                            let out_row_hi = tmem_row as usize + row_within_8 + 8; // 8 to stagger by extra 8 rows
                            let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_hi * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);
                        }

                        col_block += 1;
                    }

                    tmem_row_offset += 16;
                }

                // --- Stage 3: SMEM -> Global ---
                // Grid is (M / BM, N / BN, 1)
                // Block is (192, 1, 1)
                // SMEM_OUT is BM x BN
                // we are moving packed bf16 as u32, so have 1/2 as many columns
                const PER_WARP_BOUNDS: usize = BM * (BN / 2) / 4;
                const WIDTH: usize = (N / 2) as usize;
                const TILE_WIDTH: usize = BN / 2;
                let tile_row_base = tile_m as usize * BM;
                let tile_col_base = tile_n as usize * (BN / 2);
                let base_row = warp_id as usize * 32;

                // interate over the SMEM out linearly for coalesced loads
                let mut local_idx = lane_id as usize;
                #[unroll]
                while local_idx < PER_WARP_BOUNDS {
                    let local_row = local_idx / TILE_WIDTH;
                    let local_col = local_idx % TILE_WIDTH;
                    let smem_idx = (base_row + local_row) * 64 + local_col;

                    let global_row = tile_row_base + base_row + local_row;
                    let global_col = tile_col_base + local_col;
                    let global_idx = global_row * WIDTH + global_col;

                    unsafe {
                        *c.get_unchecked_mut(global_idx) = SMEM_OUT[smem_idx];
                    }
                    local_idx += 32;
                }

                empty_bar.arrive();
                epilogue_tile_iter += 1;
            }
        }

        // --- Stage 4: Cleanup ---
        sync_threads();
        unsafe {
            // dealloc tmem
            let _dead = tmem.dealloc();
            // dealloc barriers
            let _tma_bar_0 = tma_bar_0.inval();
            let _tma_bar_1 = tma_bar_1.inval();
            let _tma_bar_2 = tma_bar_2.inval();
            let _tma_bar_3 = tma_bar_3.inval();
            let _mma_bar_0 = mma_bar_0.inval();
            let _mma_bar_1 = mma_bar_1.inval();
            let _mma_bar_2 = mma_bar_2.inval();
            let _mma_bar_3 = mma_bar_3.inval();
            let _accum_full_0 = accum_full_0.inval();
            let _accum_full_1 = accum_full_1.inval();
            let _accum_empty_0 = accum_empty_0.inval();
            let _accum_empty_1 = accum_empty_1.inval();
            let _tile_bar = tile_bar.inval();
            let _clc_bar = clc_bar.inval();
        }
    }

    // Warp Specialized Loop:
    // moving from 4 warps to 6. removing sync_threads.
    // warps 0-3 do the epilogue
    // warp 4 exclusively handles TMA
    // warp 5 exclusively handles MMA
    // Launch dims:
    // grid dim     = (M / BM, N / BN, 1)
    // block dim    = (192, 1, 1)
    #[kernel]
    pub fn gemm_warp_specialized(
        a: *const TmaDescriptor,
        b: *const TmaDescriptor, // reminder, B should be transposed to be NxK first
        mut c: DisjointSlice<u32>,
    ) {
        // SMEM tiles contain f16/bf16 and are shape:
        // A: BM x BK
        // B: BK x BN
        // MMA instructions process MMA_K=16 at a time, so we have BK/MMA_K (4) MMA instructions
        const BF16_SIZE: usize = 2;
        const A_TILE_BYTES: usize = BM * BK * BF16_SIZE; // * 2 bc f16 is 2x u8
        const B_TILE_BYTES: usize = BK * BN * BF16_SIZE;
        const COMBINED_BYTES: u32 = (A_TILE_BYTES + B_TILE_BYTES) as u32;

        // setup smem
        static mut A0_TILE: SharedArray<u8, A_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut B0_TILE: SharedArray<u8, B_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut A1_TILE: SharedArray<u8, A_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut B1_TILE: SharedArray<u8, B_TILE_BYTES, 128> = SharedArray::UNINIT;
        // We'll also need an output SMEM now, BM x BN, these are bf16 packed into u32

        const OUTPUT_SIZE: usize = BM * BN / BF16_SIZE;
        static mut SMEM_OUT: SharedArray<u32, OUTPUT_SIZE, 128> = SharedArray::UNINIT;
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
        static mut TMA_BAR_0: Barrier = Barrier::UNINIT;
        static mut TMA_BAR_1: Barrier = Barrier::UNINIT;
        static mut MMA_BAR_0: Barrier = Barrier::UNINIT;
        static mut MMA_BAR_1: Barrier = Barrier::UNINIT;
        static mut COMPUTE_BAR: Barrier = Barrier::UNINIT;
        const TMA_WARP: u32 = 4;
        const MMA_WARP: u32 = 5;

        // TMA will Swizzle/copy BM/BN x BK f16 tiles from GMEM to SMEM the
        // swizzling is so that the matrices are laid out correctly for MMA.
        //
        // So LBO, or the byte distance along K dim to the neighbor core
        // matrix, is 16b (hardcoded row size of a core matrix)
        // and SBO, byte distance to the neighbor core matrix along the
        // strided dim, is 1024b, because we have 8 K-groups with BK=64, so
        // the 8 K-groups for a row of size 128b, then each core matrix has
        // 8 rows, so the next row of core matrices is 8 x 128b = 1024b.
        const LBO_BYTES: u32 = 16;
        const SBO_BYTES: u32 = 1024;

        // live info:
        let tid = thread::threadIdx_x();
        let warp_id = warp::warp_id();
        let lane_id = tid % 32;
        let thread0 = tid == 0;

        let tile_m = thread::blockIdx_x(); // which BM-row block of C
        let tile_n = thread::blockIdx_y(); // which BM-col block of C
        let m_offset = (tile_m as usize * BM) as i32;
        let n_offset = (tile_n as usize * BN) as i32;

        // --- Initialize Barriers and TMEM ---
        let tma_bar_0 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR_0);
        let tma_bar_1 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR_1);
        let mma_bar_0 = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR_0);
        let mma_bar_1 = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR_1);
        let compute_bar = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut COMPUTE_BAR);

        let tma_bar_0 = unsafe { tma_bar_0.init(1) };
        let tma_bar_1 = unsafe { tma_bar_1.init(1) };
        let mma_bar_0 = unsafe { mma_bar_0.init(1) };
        let mma_bar_1 = unsafe { mma_bar_1.init(1) };
        let compute_bar = unsafe { compute_bar.init(1) };
        unsafe { fence_proxy_async_shared_cta() };
        thread::sync_threads();

        // pre signal mma bars so they don't stall on the first iter:
        if thread0 {
            mma_bar_0.arrive();
            mma_bar_1.arrive();
        }
        sync_threads();

        let tmem =
            TmemGuard::<TmemUninit, { BN as u32 }>::from_static(&raw mut TMEM_ADDR as *mut u32);
        let tmem = unsafe { tmem.alloc() };
        thread::sync_threads();

        let mma_desc = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M128_N128)
            .element_type(Tcgen05ElementType::BF16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();

        // --- Stage 1: K-Loop ---
        // Warp Specialized Version
        // Warp 4: TMA Producer loads tiles
        // Warp 5: MMA Consumer computes tiles
        // Warp 0-3: Idle

        let k_iters = (K / BK) as u32;

        if warp_id == TMA_WARP {
            let lane0 = warp::lane_id() == 0;
            let mut k_idx = 0u32;

            while k_idx < k_iters {
                let phase = k_idx & 1;
                let parity = (k_idx >> 1) & 1;

                let (smem_a, smem_b, mma_bar, tma_bar) = match phase {
                    0 => (
                        &raw mut A0_TILE as *mut u8,
                        &raw mut B0_TILE as *mut u8,
                        &mma_bar_0,
                        &tma_bar_0,
                    ),
                    _ => (
                        &raw mut A1_TILE as *mut u8,
                        &raw mut B1_TILE as *mut u8,
                        &mma_bar_1,
                        &tma_bar_1,
                    ),
                };

                // Wait for MMA computation to finish, freeing the tile.
                while !mma_bar.try_wait_parity(parity) {}

                if lane0 {
                    let k_offset = (k_idx as usize * BK) as i32;
                    unsafe {
                        cp_async_bulk_tensor_2d_g2s(
                            smem_a,
                            a,
                            k_offset,
                            m_offset,
                            tma_bar.as_ptr() as *mut Barrier,
                        );
                        cp_async_bulk_tensor_2d_g2s(
                            smem_b,
                            b,
                            k_offset,
                            n_offset,
                            tma_bar.as_ptr() as *mut Barrier,
                        );
                    }
                    tma_bar.arrive_expect_tx(COMBINED_BYTES);
                }
                k_idx += 1;
            }
        }

        if warp_id == MMA_WARP {
            let lane0 = warp::lane_id() == 0;
            let mut k_idx = 0u32;

            while k_idx < k_iters {
                let phase = k_idx & 1;
                let parity = (k_idx >> 1) & 1;
                let is_end = k_idx + 1 == k_iters;

                let (smem_a, smem_b, mma_bar, tma_bar) = match phase {
                    0 => (
                        &raw mut A0_TILE as u64,
                        &raw mut B0_TILE as u64,
                        &mma_bar_0,
                        &tma_bar_0,
                    ),
                    _ => (
                        &raw mut A1_TILE as u64,
                        &raw mut B1_TILE as u64,
                        &mma_bar_1,
                        &tma_bar_1,
                    ),
                };

                // wait for tile memory to be filled
                while !tma_bar.try_wait_parity(parity) {}

                if lane0 {
                    let mut k_sub = 0;
                    while k_sub < BK as u64 / 16 {
                        let k_offset = (k_sub * 32) as u64;
                        let a_desc = Tcgen05SmemDescriptor::from_bytes(
                            smem_a + k_offset,
                            LBO_BYTES,
                            SBO_BYTES,
                            Tcgen05SwizzleMode::Swizzle128B,
                        );
                        let b_desc = Tcgen05SmemDescriptor::from_bytes(
                            smem_b + k_offset,
                            LBO_BYTES,
                            SBO_BYTES,
                            Tcgen05SwizzleMode::Swizzle128B,
                        );

                        // accumulate into d always except for the very first time.
                        let accumlate = k_idx > 0 || k_sub > 0;
                        unsafe {
                            tcgen05_mma_f16(
                                tmem.address().raw(),
                                a_desc.raw(),
                                b_desc.raw(),
                                mma_desc.raw(),
                                accumlate,
                            )
                        };
                        k_sub += 1;
                    }

                    unsafe {
                        if is_end {
                            tcgen05_commit_shared_cluster(&raw mut COMPUTE_BAR as *mut u64);
                        } else {
                            tcgen05_commit_shared_cluster(mma_bar.as_ptr() as *mut u64);
                        }
                    }
                }
                k_idx += 1;
            }
        }

        // --- Stage 2: TMEM -> Registers -> SMEM ---
        if warp_id < 4 {
            while !compute_bar.try_wait_parity(0u32) {}
            // first 4 warps loads and then moves to smem
            // TMEM is BM x BN (128 x 128)
            // values are f32
            // tcgen05_ld_16x256b_pure loads 4x f32 per thread
            // loads 16 rows x 8 columns
            // so we load 2x matrices with a col offset of 16
            // when we pass the address its packed (row << 16) | column
            // so the addr is tmem.address + (tmem_row << 16) | col_offset
            // lines up exactly with stmatrix_m8n8_x2
            let warp_row = (warp_id * 32) as usize;
            let row_stride_bytes = BN * 2;
            let col_step = 16;

            // you are probably thinking:
            // "where did you get these numbers? it feels like they dont make any
            // sense?"
            // and you would be correct, instead of each lane actually computing
            // real addresses, each lane(thread) holds data, but then only lanes
            // 0-15 provide addresses.
            // The load command we are using (stmatrix_m8n8_x2) takes 2 8x8
            // matrices, it is designed to match the output from the tmem loads we
            // used. with that in mind lanes 0-7 compute the row addresses for the
            // first matrix, and lanes 8-15 compute the row addresses for the
            // second.
            let second_load_offset = 8; // high columns
            let row_within_8 = (lane_id % 8) as usize;
            let is_second_matrix = (8..16).contains(&lane_id);
            let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

            let mut tmem_row_offset = 0u32;
            while tmem_row_offset < 32 {
                let tmem_row = warp_row as u32 + tmem_row_offset;

                let mut col_block = 0u32;
                while col_block < 8 {
                    let col_offset = (col_block * col_step) as usize;
                    unsafe {
                        let regs_a = tcgen05_ld_16x256b_pure(
                            tmem.address().raw() + (tmem_row << 16) + col_offset as u32,
                        );
                        tcgen05_load_wait();
                        let regs_b = tcgen05_ld_16x256b_pure(
                            tmem.address().raw()
                                + (tmem_row << 16)
                                + col_offset as u32
                                + second_load_offset,
                        );
                        tcgen05_load_wait();

                        let p0_lo = cvt_f32x2_bf16x2(regs_a.x(), regs_a.y());
                        let p1_lo = cvt_f32x2_bf16x2(regs_b.x(), regs_b.y());

                        let out_row_lo = tmem_row as usize + row_within_8;
                        let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                            out_row_lo * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                        );
                        stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                        let p0_hi = cvt_f32x2_bf16x2(regs_a.z(), regs_a.w());
                        let p1_hi = cvt_f32x2_bf16x2(regs_b.z(), regs_b.w());
                        let out_row_hi = tmem_row as usize + row_within_8 + 8; // 8 to stagger by extra 8 rows
                        let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                            out_row_hi * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                        );
                        stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);
                    }

                    col_block += 1;
                }

                tmem_row_offset += 16;
            }
        }

        sync_threads();

        // --- Stage 3: SMEM -> Global ---
        // Grid is (M / BM, N / BN, 1)
        // Block is (192, 1, 1)
        // SMEM_OUT is BM x BN
        // we are moving packed bf16 as u32, so have 1/2 as many columns
        let width = (N / 2) as usize;
        let tile_row_base = tile_m as usize * BM;
        let tile_col_base = tile_n as usize * (BN / 2);

        // interate over the SMEM out linearly for coalesced loads
        let mut local_idx = tid as usize;
        while local_idx < BM * (BN / 2) {
            // row is the top local_idx / width
            let local_row = local_idx >> 6;
            // col is only the last 6 bits, equiv to local_idx % width
            let local_col = local_idx & 63;

            let global_row = tile_row_base + local_row;
            let global_col = tile_col_base + local_col;
            let global_idx = global_row * width + global_col;

            unsafe {
                *c.get_unchecked_mut(global_idx) = SMEM_OUT[local_idx];
            }
            local_idx += 192;
        }

        // --- Stage 4: Cleanup ---
        sync_threads();
        unsafe {
            // dealloc tmem
            let _dead = tmem.dealloc();
            let _tma_bar0 = tma_bar_0.inval();
            let _tma_bar1 = tma_bar_1.inval();
            let _mma_bar0 = mma_bar_0.inval();
            let _mma_bar1 = mma_bar_1.inval();
            let _compute_bar = compute_bar.inval();
        }
    }

    // Launch dims:
    // grid dim     = (M / BM, N / BN, 1)
    // block dim    = (128, 1, 1)
    #[kernel]
    pub fn gemm_tcgen05_basic(
        a: *const TmaDescriptor,
        b: *const TmaDescriptor, // reminder, B should be transposed to be NxK first
        mut c: DisjointSlice<u32>,
    ) {
        // SMEM tiles contain f16/bf16 and are shape:
        // A: BM x BK
        // B: BK x BN
        // MMA instructions process MMA_K=16 at a time, so we have BK/MMA_K (4) MMA instructions
        const BF16_SIZE: usize = 2;
        const A_TILE_BYTES: usize = BM * BK * BF16_SIZE; // * 2 bc f16 is 2x u8
        const B_TILE_BYTES: usize = BK * BN * BF16_SIZE;
        const COMBINED_BYTES: u32 = (A_TILE_BYTES + B_TILE_BYTES) as u32;
        static mut TILE_A: SharedArray<u8, A_TILE_BYTES, 128> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<u8, B_TILE_BYTES, 128> = SharedArray::UNINIT;
        // We'll also need an output SMEM now, BM x BN, these are bf16 packed into u32
        const OUTPUT_SIZE: usize = BM * BN / BF16_SIZE;
        static mut SMEM_OUT: SharedArray<u32, OUTPUT_SIZE, 128> = SharedArray::UNINIT;
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
        static mut TMA_BAR: Barrier = Barrier::UNINIT;
        static mut MMA_BAR: Barrier = Barrier::UNINIT;

        // TMA will Swizzle/copy BM/BN x BK f16 tiles from GMEM to SMEM the
        // swizzling is so that the matrices are laid out correctly for MMA.
        //
        // So LBO, or the byte distance along K dim to the neighbor core
        // matrix, is 16b (hardcoded row size of a core matrix)
        // and SBO, byte distance to the neighbor core matrix along the
        // strided dim, is 1024b, because we have 8 K-groups with BK=64, so
        // the 8 K-groups for a row of size 128b, then each core matrix has
        // 8 rows, so the next row of core matrices is 8 x 128b = 1024b.
        const LBO_BYTES: u32 = 16;
        const SBO_BYTES: u32 = 1024;
        let swizzle = Tcgen05SwizzleMode::Swizzle128B;

        // live info:
        let tid = thread::threadIdx_x();
        let warp_id = warp::warp_id();
        let lane_id = tid % 32;
        let thread0 = tid == 0;

        let tile_m = thread::blockIdx_x(); // which BM-row block of C
        let tile_n = thread::blockIdx_y(); // which BM-col block of C
        let m_offset = (tile_m as usize * BM) as i32;
        let n_offset = (tile_n as usize * BN) as i32;

        // --- Initialize Barriers and TMEM ---
        let tma_bar = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR);
        let mma_bar = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR);

        let tma_bar = unsafe { tma_bar.init(thread::blockDim_x()) };
        let mma_bar = unsafe { mma_bar.init(1) };
        let mut tma_token;
        unsafe { fence_proxy_async_shared_cta() };
        thread::sync_threads();

        let tmem =
            TmemGuard::<TmemUninit, { BN as u32 }>::from_static(&raw mut TMEM_ADDR as *mut u32);
        let tmem = unsafe { tmem.alloc() };
        thread::sync_threads();

        let mma_desc = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M128_N128)
            .element_type(Tcgen05ElementType::BF16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();

        // Stage 1: K-Loop
        let k_iters = (K / BK) as u32;
        let mut k_idx: u32 = 0;

        while k_idx < k_iters {
            let phase = k_idx & 1; // % 2

            // TMA Load
            // load the whole 128x64 tile at once.
            if thread0 {
                let k_base = (k_idx as usize * BK) as i32;
                unsafe {
                    cp_async_bulk_tensor_2d_g2s(
                        &raw mut TILE_A as *mut u8,
                        a,
                        k_base,
                        m_offset,
                        &raw mut TMA_BAR,
                    );
                    cp_async_bulk_tensor_2d_g2s(
                        &raw mut TILE_B as *mut u8,
                        b,
                        k_base,
                        n_offset,
                        &raw mut TMA_BAR,
                    );
                }
                tma_token = tma_bar.arrive_expect_tx(COMBINED_BYTES);
            } else {
                tma_token = tma_bar.arrive();
            }

            tma_bar.wait(tma_token);

            // --- MMAs within the Tile ---
            // each MMA can consume K=16 at a time (two
            // k groups) LBO = 16, so 2xLBO is 32 bytes, thats the offset
            // between each K=16
            if thread0 {
                let smem_a_ptr = &raw const TILE_A as u64;
                let smem_b_ptr = &raw const TILE_B as u64;

                let mut k_sub = 0;
                while k_sub < BK as u64 / 16 {
                    let k_offset = (k_sub * 32) as u64;
                    let a_desc = Tcgen05SmemDescriptor::from_bytes(
                        smem_a_ptr + k_offset,
                        LBO_BYTES,
                        SBO_BYTES,
                        swizzle,
                    );
                    let b_desc = Tcgen05SmemDescriptor::from_bytes(
                        smem_b_ptr + k_offset,
                        LBO_BYTES,
                        SBO_BYTES,
                        swizzle,
                    );

                    // accumulate into d always except for the very first time.
                    let accumlate = k_idx > 0 || k_sub > 0;
                    unsafe {
                        tcgen05_mma_f16(
                            tmem.address().raw(),
                            a_desc.raw(),
                            b_desc.raw(),
                            mma_desc.raw(),
                            accumlate,
                        )
                    };
                    k_sub += 1;
                }

                unsafe {
                    tcgen05_commit_shared_cluster(&raw mut MMA_BAR as *mut u64);
                }
            }

            while !mma_bar.try_wait_parity(phase) {} // all threads wait
            sync_threads();
            k_idx += 1;
        }

        // --- Stage 2: TMEM -> Registers -> SMEM ---

        // each warp loads and then moves to smem
        // TMEM is BM x BN (128 x 128)
        // values are f32
        // tcgen05_ld_16x256b_pure loads 4x f32 per thread
        // loads 16 rows x 8 columns
        // so we load 2x matrices with a col offset of 16
        // when we pass the address its packed (row << 16) | column
        // so the addr is tmem.address + (tmem_row << 16) | col_offset
        // lines up exactly with stmatrix_m8n8_x2
        let warp_row = (warp_id * 32) as usize;
        let row_stride_bytes = BN * 2;
        let col_step = 16;

        // you are probably thinking:
        // "where did you get these numbers? it feels like they dont make any
        // sense?"
        // and you would be correct, instead of each lane actually computing
        // real addresses, each lane(thread) holds data, but then only lanes
        // 0-15 provide addresses.
        // The load command we are using (stmatrix_m8n8_x2) takes 2 8x8
        // matrices, it is designed to match the output from the tmem loads we
        // used. with that in mind lanes 0-7 compute the row addresses for the
        // first matrix, and lanes 8-15 compute the row addresses for the
        // second.
        let second_load_offset = 8; // high columns
        let row_within_8 = (lane_id % 8) as usize;
        let is_second_matrix = (8..16).contains(&lane_id);
        let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

        let mut tmem_row_offset = 0u32;
        while tmem_row_offset < 32 {
            let tmem_row = warp_row as u32 + tmem_row_offset;

            let mut col_block = 0u32;
            while col_block < 8 {
                let col_offset = (col_block * col_step) as usize;
                unsafe {
                    let regs_a = tcgen05_ld_16x256b_pure(
                        tmem.address().raw() + (tmem_row << 16) + col_offset as u32,
                    );
                    tcgen05_load_wait();
                    let regs_b = tcgen05_ld_16x256b_pure(
                        tmem.address().raw()
                            + (tmem_row << 16)
                            + col_offset as u32
                            + second_load_offset,
                    );
                    tcgen05_load_wait();

                    let p0_lo = cvt_f32x2_bf16x2(regs_a.x(), regs_a.y());
                    let p1_lo = cvt_f32x2_bf16x2(regs_b.x(), regs_b.y());

                    let out_row_lo = tmem_row as usize + row_within_8;
                    let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                        out_row_lo * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                    );
                    stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                    let p0_hi = cvt_f32x2_bf16x2(regs_a.z(), regs_a.w());
                    let p1_hi = cvt_f32x2_bf16x2(regs_b.z(), regs_b.w());
                    let out_row_hi = tmem_row as usize + row_within_8 + 8; // 8 to stagger by extra 8 rows
                    let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                        out_row_hi * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                    );
                    stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);
                }

                col_block += 1;
            }

            tmem_row_offset += 16;
        }
        sync_threads();

        // --- Stage 3: SMEM -> Global ---
        // Grid is (M / BM, N / BN, 1)
        // Block is (128, 1, 1)
        // SMEM_OUT is BM x BN
        // we are moving packed bf16 as u32, so have 1/2 as many columns
        let width = (N / 2) as usize;
        let tile_row_base = tile_m as usize * BM;
        let tile_col_base = tile_n as usize * (BN / 2);

        // interate over the SMEM out linearly for coalesced loads
        let mut local_idx = tid as usize;
        while local_idx < BM * (BN / 2) {
            // row is the top local_idx / width
            let local_row = local_idx >> 6;
            // col is only the last 6 bits, equiv to local_idx % width
            let local_col = local_idx & 63;

            let global_row = tile_row_base + local_row;
            let global_col = tile_col_base + local_col;
            let global_idx = global_row * width + global_col;

            unsafe {
                *c.get_unchecked_mut(global_idx) = SMEM_OUT[local_idx];
            }
            local_idx += 128;
        }

        // --- Stage 4: Cleanup ---
        sync_threads();
        // dealloc tmem
        let _dead = unsafe { tmem.dealloc() };
        let _tma_bar = unsafe { tma_bar.inval() };
        let _mma_bar = unsafe { mma_bar.inval() };
    }

    #[kernel]
    pub fn gemm_tma_pipeline(
        a: *const TmaDescriptor,
        b: *const TmaDescriptor,
        mut c: DisjointSlice<f32, thread::Index2D<N>>,
    ) {
        // Frozen copies of the tuning consts (shadowing the module-level ones)
        // so this kernel stays intact when they change for the next kernel.
        // M/N/K stay module-level: N is baked into the signature above.
        const BK: usize = 16;
        const BN: usize = 16; // block is BN×BM threads
        const BM: usize = 16; // block is BN×BM threads
        const T: usize = 8; // thread computes TTILExTTILE
        const A_SIZE: usize = BM * BK * T;
        const B_SIZE: usize = BK * BN * T;
        const A_STRIDE: usize = BK;
        const B_STRIDE: usize = BN * T;
        const TILE_BYTES: u32 = ((A_SIZE + B_SIZE) * 4) as u32; // 4 is f32 bytes.

        // row-major indexing: a[row*K + k], b[k*N + col], c[row*N + col].
        // Bounds are exact multiples of TILE
        static mut TILE_A_1: SharedArray<f32, A_SIZE, 128> = SharedArray::UNINIT;
        static mut TILE_B_1: SharedArray<f32, B_SIZE, 128> = SharedArray::UNINIT;
        static mut TILE_A_2: SharedArray<f32, A_SIZE, 128> = SharedArray::UNINIT;
        static mut TILE_B_2: SharedArray<f32, B_SIZE, 128> = SharedArray::UNINIT;

        static mut BAR_1: Barrier = Barrier::UNINIT;
        static mut BAR_2: Barrier = Barrier::UNINIT;
        let mut token_1: u64;
        let mut token_2: u64;

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
                mbarrier_init(&raw mut BAR_1, thread::blockDim_x() * thread::blockDim_y());
                mbarrier_init(&raw mut BAR_2, thread::blockDim_x() * thread::blockDim_y());
                fence_proxy_async_shared_cta();
            }
        }
        sync_threads();

        let mut t = 0usize;
        // pre-load first tile with TMA
        // SAFETY: only tid = 0 issues the load.
        if tid == 0 {
            unsafe {
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut TILE_A_1 as *mut u8,
                    a,
                    (t * BK) as i32,
                    row as i32,
                    &raw mut BAR_1,
                );
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut TILE_B_1 as *mut u8,
                    b,
                    col as i32,
                    (t * BK) as i32,
                    &raw mut BAR_1,
                );
            }
        }
        // all threads arrive; the issuing thread also registers the
        // expected TMA bytes for this phase.
        token_1 = unsafe {
            if tid == 0 {
                mbarrier_arrive_expect_tx(&raw mut BAR_1, 1, TILE_BYTES)
            } else {
                mbarrier_arrive(&raw mut BAR_1)
            }
        };

        while t < K / BK {
            // start the next load.
            t += 1;
            if tid == 0 {
                unsafe {
                    cp_async_bulk_tensor_2d_g2s(
                        &raw mut TILE_A_2 as *mut u8,
                        a,
                        (t * BK) as i32,
                        row as i32,
                        &raw mut BAR_2,
                    );
                    cp_async_bulk_tensor_2d_g2s(
                        &raw mut TILE_B_2 as *mut u8,
                        b,
                        col as i32,
                        (t * BK) as i32,
                        &raw mut BAR_2,
                    );
                }
            }
            token_2 = unsafe {
                if tid == 0 {
                    mbarrier_arrive_expect_tx(&raw mut BAR_2, 1, TILE_BYTES)
                } else {
                    mbarrier_arrive(&raw mut BAR_2)
                }
            };
            // wait for pipeline stage 1 load to finish,, NOT the load we just started.
            unsafe { while !mbarrier_try_wait(&raw const BAR_1, token_1) {} }

            // Compute with pipeline stage 1
            let mut k = 0usize;
            #[unroll]
            while k < BK {
                // load from smem to registers for computation
                let mut l = 0usize;
                #[unroll]
                while l < T {
                    unsafe {
                        reg_a[l] = TILE_A_1[(r0 + l) * A_STRIDE + k];
                        reg_b[l] = TILE_B_1[k * B_STRIDE + c0 + l];
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

            unsafe {
                fence_proxy_async_shared_cta();
            }
            sync_threads();

            // launch tma load for pipeline stage 1
            t += 1;
            if t < K / BK {
                if tid == 0 {
                    unsafe {
                        cp_async_bulk_tensor_2d_g2s(
                            &raw mut TILE_A_1 as *mut u8,
                            a,
                            (t * BK) as i32,
                            row as i32,
                            &raw mut BAR_1,
                        );
                        cp_async_bulk_tensor_2d_g2s(
                            &raw mut TILE_B_1 as *mut u8,
                            b,
                            col as i32,
                            (t * BK) as i32,
                            &raw mut BAR_1,
                        );
                    }
                }
                token_1 = unsafe {
                    if tid == 0 {
                        mbarrier_arrive_expect_tx(&raw mut BAR_1, 1, TILE_BYTES)
                    } else {
                        mbarrier_arrive(&raw mut BAR_1)
                    }
                };
            }

            // wait for second pipeline stage.
            unsafe { while !mbarrier_try_wait(&raw const BAR_2, token_2) {} }

            // now perform computations with pipeline stage 2
            let mut k = 0usize;
            #[unroll]
            while k < BK {
                // load from smem to registers for computation
                let mut l = 0usize;
                #[unroll]
                while l < T {
                    unsafe {
                        reg_a[l] = TILE_A_2[(r0 + l) * A_STRIDE + k];
                        reg_b[l] = TILE_B_2[k * B_STRIDE + c0 + l];
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
            unsafe {
                fence_proxy_async_shared_cta();
            }
            sync_threads();
        }

        // === computations finished ===
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

        thread::sync_threads();
        if tid == 0 {
            unsafe {
                mbarrier_inval(&raw mut BAR_1);
                mbarrier_inval(&raw mut BAR_2);
            }
        }
    }

    #[kernel]
    pub fn gemm_register(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, thread::Index2D<N>>) {
        // Frozen copies of the tuning consts (shadowing the module-level ones)
        // so this kernel stays intact when they change for the next kernel.
        const BK: usize = 16;
        const BN: usize = 16; // block is BN×BM threads
        const BM: usize = 16; // block is BN×BM threads
        const T: usize = 8; // thread computes TTILExTTILE
        const A_SIZE: usize = BM * BK * T;
        const B_SIZE: usize = BK * BN * T;
        const A_STRIDE: usize = BK;
        const B_STRIDE: usize = BN * T;

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
// Host-side GEMM driver for the tcgen05 kernel.
//
// The kernel consumes bf16 inputs through 128B-swizzled TMA descriptors: A is
// M×K row-major, and B must be pre-transposed to N×K (see `transpose`) so both
// descriptors are K-major. C comes back as M×N bf16 packed two-per-u32, the
// layout the stmatrix epilogue produces. `c` is allocated per call (zeroed, so
// an unfinished kernel reads as all-zeros rather than garbage) and returned on
// the device for the caller to copy down or time.
use half::bf16;
use kernels::{BK, BM, BN, K, LoadedModule, M, N};

/// Round f32 host data to bf16, the kernel's input precision.
pub fn to_bf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

/// Unpack the kernel's C buffer (two bf16 per u32, low half first) to f32.
pub fn from_packed_bf16(v: &[u32]) -> Vec<f32> {
    v.iter()
        .flat_map(|&p| {
            [
                bf16::from_bits(p as u16).to_f32(),
                bf16::from_bits((p >> 16) as u16).to_f32(),
            ]
        })
        .collect()
}

/// Transpose a row-major `rows`×`cols` matrix.
pub fn transpose<T: Copy>(v: &[T], rows: usize, cols: usize) -> Vec<T> {
    (0..cols)
        .flat_map(|col| (0..rows).map(move |row| v[row * cols + col]))
        .collect()
}

/// Create a TMA tensor map descriptor for a 2D bf16 tensor with SWIZZLE_128B.
///
/// TMA applies the 128-byte XOR swizzle during the GMEM→SMEM copy, so the
/// kernel's SMEM descriptors (built with `Swizzle128B`) see core matrices at
/// the MMA's hardcoded strides. The swizzle needs the box inner dim to span
/// exactly 128 bytes: BK (64) bf16 elements.
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
    let global_strides: [u64; 1] = [width * std::mem::size_of::<bf16>() as u64];
    let box_dim: [u32; 2] = [tile_width, tile_height];
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            tensor_rank,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != cuda_sys::cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled failed: {:?}", result).into());
    }

    Ok(unsafe { tensor_map.assume_init() })
}

/// Create a plain (unswizzled) TMA tensor map descriptor for a 2D f32
/// tensor. Used by the pre-tcgen05 `gemm_tma_pipeline` kernel, which copies
/// row-major f32 tiles verbatim.
pub fn create_tma_descriptor_f32(
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
        return Err(format!("cuTensorMapEncodeTiled (f32) failed: {:?}", result).into());
    }

    Ok(unsafe { tensor_map.assume_init() })
}

/// One CTA per BM×BN output tile, 192 threads (6 warps) per CTA. The CLC
/// kernel wants a *linear* grid — it decodes tile_m = ctaid % TILES_M,
/// tile_n = ctaid / TILES_M (column major) and steals work in blocks of 4
/// consecutive ctaids, matching its `#[cluster_launch(4, 1, 1)]`. The
/// generated host stub applies the cluster dims itself, so only the grid
/// shape lives here.
fn launch() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (((M / BM) * (N / BN)) as u32, 1, 1),
        // 6 warps: 0-3 epilogue, 4 TMA producer, 5 MMA consumer.
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Device-side launch state for `C = A · B`: the TMA descriptors and output
/// buffer, built once so repeated launches (benchmarking) time only the
/// kernel — not two descriptor uploads and a 16 MB allocation per call.
///
/// `a` is M×K bf16 row-major; `b_t` is B pre-transposed to N×K bf16
/// row-major, so its TMA descriptor is K-major like A's. `c` is M×N bf16
/// packed two-per-u32 (M·N/2 u32s; unpack with [`from_packed_bf16`]).
pub struct Gemm {
    dev_a_map: DeviceBuffer<u64>,
    dev_b_map: DeviceBuffer<u64>,
    pub c: DeviceBuffer<u32>,
}

impl Gemm {
    pub fn new(
        stream: &CudaStream,
        a: &DeviceBuffer<bf16>,
        b_t: &DeviceBuffer<bf16>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Zeroed only so an unfinished kernel reads as all-zeros rather than
        // garbage; a completed launch writes every element.
        let c = DeviceBuffer::<u32>::zeroed(stream, M * N / 2)?;
        let a_map = create_tma_descriptor(
            a.cu_deviceptr() as *mut std::ffi::c_void,
            K as u64,
            M as u64,
            BK as u32,
            BM as u32,
        )?;
        let dev_a_map = DeviceBuffer::from_host(stream, &a_map.opaque[..])?;
        let b_map = create_tma_descriptor(
            b_t.cu_deviceptr() as *mut std::ffi::c_void,
            K as u64,
            N as u64,
            BK as u32,
            BN as u32,
        )?;
        let dev_b_map = DeviceBuffer::from_host(stream, &b_map.opaque[..])?;
        Ok(Self {
            dev_a_map,
            dev_b_map,
            c,
        })
    }

    pub fn launch(
        &mut self,
        stream: &CudaStream,
        module: &LoadedModule,
    ) -> Result<(), Box<dyn std::error::Error>> {
        module.gemm(
            stream,
            launch(),
            self.dev_a_map.cu_deviceptr() as *const TmaDescriptor,
            self.dev_b_map.cu_deviceptr() as *const TmaDescriptor,
            &mut self.c,
        )?;
        Ok(())
    }
}

/// One-shot convenience for correctness checks: build the descriptors,
/// launch once, return C.
pub fn matmul(
    stream: &CudaStream,
    module: &LoadedModule,
    a: &DeviceBuffer<bf16>,
    b_t: &DeviceBuffer<bf16>,
) -> Result<DeviceBuffer<u32>, Box<dyn std::error::Error>> {
    let mut gemm = Gemm::new(stream, a, b_t)?;
    gemm.launch(stream, module)?;
    Ok(gemm.c)
}

// BASELINE
// cuBLAS SGEMM  (fill in once measured)

// BENCHMARKS:
// -- GPU (B200), 8192×8192×8192, bench_all (same shape/data, fair ladder):
// WORK STEALING (WAVES=4)      1.5600 ms   704.8 TFLOP/s
// WORK STEALING (WAVES=2)      2.2133 ms   496.8 TFLOPs
// TCGEN05 v3 WARP SPECIALIZED  2.1693 ms   506.8 TFLOPs    (36% of ~1400 bf16 SoL)
// TCGEN05 v2 PIPELINE          3.7748 ms   291.3 TFLOPs
// TCGEN05 v1                   4.0277 ms   273.0 TFLOPs    (19.5% of ~1400 bf16 SoL)
// TMA PIPELINE(=2)             23.0041 ms  47.8 TFLOPs
// REGISTER TILES (B=16,T=8):   36.9776 ms  29.7 TFLOPs
// NAIVE:                       164.7083 ms 6.7 TFLOPs
// (SHARED+TILES unlaunchable at module BK=64: needs a local BK=32 freeze)
//
// -- GPU (B200), 8192×1024×4096 (old shape; timing included per-call
//    alloc + descriptor uploads — superseded by the numbers above):
// TCGEN05 v1                   0.8972 ms   76.6 TFLOPs     102.8 GB/s
// TMA PIPELINE(=2)                         29.3 TFLOPs
// -- GPU (H100), 8192×1024×4096:
// TMA PIPELINE(=2)             2.4390 ms   28175.0 GFLOPs  75.7 GB/s (min traffic)
// REGISTER TILES (B=32,T=4):   2.6268 ms   26161.1 GFLOPs  70.3 GB/s
// REGISTER TILES (B=8,T=8):    3.9705 ms   17307.6 GFLOPs  46.5 GB/s
// SHARED+TILES:                8.5085 ms   8076.5 GFLOPs   21.7 GB/s
// NAIVE:                       11.1397 ms  6168.9 GFLOPs   16.6 GB/s
//
// tuning sweep (BM=BN=BK required equal by the fill pattern; T = thread tile):
//   B=8  T=8:  17.3 TFLOP/s  (64 thr)      B=16 T=8: 13.6  (256 thr, reg-bound)
//   B=16 T=4:  16.1           (256 thr)    B=32 T=4: 26.2  (1024 thr) <- best
//   B=32 T=4 + #[unroll(2)] on t-loop: 21.5 (i-cache/reg pressure, reverted)
