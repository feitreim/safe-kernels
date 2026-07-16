//! Stage-selection benchmark on the REAL gemm kernel (not a microbenchmark).
//!
//! Three copies of the tcgen05 CLC work-stealing gemm from `kernels/gemm`,
//! identical except for how each pipeline stage's resources are selected:
//!
//!   `gemm_arith`: stage-indexed `SharedArray`s, stage selected by pointer
//!     arithmetic (`base.add(i)`) — the current gemm idiom, everything in
//!     registers.
//!
//!   `gemm_match`: one static per stage, selected by a `match` yielding a
//!     mixed tuple (raw smem pointers + `ManagedBarrier` references) — the
//!     kernel's old selection idiom, which stock LLVM lowers through a
//!     local-memory depot (historically 710 vs 857 TFLOP/s at 8192³).
//!
//!   `gemm_branch`: same per-stage statics, but each `match` duplicates the
//!     whole stage body into its arm — no values are ever selected; every
//!     resource is a compile-time constant inside its arm, and the stage
//!     choice is a warp-uniform branch paid in code size.
//!
//! The kernels share this module so all land in one `barrier_bench.ptx` and
//! run on identical descriptors/inputs. `src/main.rs` times each and
//! cross-checks their outputs, which must match bitwise (same accumulation
//! order, same inputs).
//!
//! Run on a GPU (via Modal):  ./run.sh barrier_bench

use cuda_core::{
    CudaStream, DeviceBuffer, LaunchConfig,
    sys::{
        self as cuda_sys, CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
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
            mbarrier_init, mbarrier_inval, mbarrier_try_wait_parity,
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

    // Stage-indexed SharedArray variant: identical to kernels/gemm `gemm`.
    // See that file for the full pipeline/work-stealing commentary; comments
    // here are trimmed to what differs between the two variants.
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub fn gemm_arith(
        a: *const TmaDescriptor,
        b: *const TmaDescriptor, // reminder, B should be transposed to be NxK first
        mut c: DisjointSlice<u32>,
    ) {
        const BF16_SIZE: usize = 2;
        const A_TILE_BYTES: usize = BM * BK * BF16_SIZE;
        const B_TILE_BYTES: usize = BK * BN * BF16_SIZE;
        const COMBINED_BYTES: u32 = (A_TILE_BYTES + B_TILE_BYTES) as u32;
        const OUTPUT_SIZE: usize = BM * BN / BF16_SIZE;
        const CLUSTER_SIZE: u32 = 4;
        const TILES_M: u32 = (M / BM) as u32;
        const STAGES: u32 = 4;
        const NUM_ACCUM_STAGES: u32 = 2;

        // tiles and pipeline barriers live in stage-indexed arrays so
        // selecting a stage is pointer arithmetic.
        static mut SMEM_A: SharedArray<u8, { A_TILE_BYTES * STAGES as usize }, 128> =
            SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, { B_TILE_BYTES * STAGES as usize }, 128> =
            SharedArray::UNINIT;
        static mut SMEM_OUT: SharedArray<u32, OUTPUT_SIZE, 128> = SharedArray::UNINIT;
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

        static mut TMA_BARS: SharedArray<u64, { STAGES as usize }, 8> = SharedArray::UNINIT;
        static mut MMA_BARS: SharedArray<u64, { STAGES as usize }, 8> = SharedArray::UNINIT;

        static mut ACCUM_FULL: SharedArray<u64, { NUM_ACCUM_STAGES as usize }, 8> =
            SharedArray::UNINIT;
        static mut ACCUM_EMPTY: SharedArray<u64, { NUM_ACCUM_STAGES as usize }, 8> =
            SharedArray::UNINIT;
        static mut TILE_READY: Barrier = Barrier::UNINIT;

        static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;
        static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
        static mut CLC_BAR: Barrier = Barrier::UNINIT;

        const SBO_BYTES: u32 = 1024;
        const LBO_BYTES: u32 = 16;
        const TMA_WARP: u32 = 4;
        const MMA_WARP: u32 = 5;

        let tid = thread::threadIdx_x();
        let warp_id = warp::warp_id();
        let lane_id = tid % 32;
        let thread0 = tid == 0;

        // stage-indexed accessors: base + i, always register arithmetic
        let tma_bar_at = |i: u32| unsafe { (&raw mut TMA_BARS as *mut Barrier).add(i as usize) };
        let mma_bar_at = |i: u32| unsafe { (&raw mut MMA_BARS as *mut Barrier).add(i as usize) };
        let accum_full_at =
            |i: u32| unsafe { (&raw mut ACCUM_FULL as *mut Barrier).add(i as usize) };
        let accum_empty_at =
            |i: u32| unsafe { (&raw mut ACCUM_EMPTY as *mut Barrier).add(i as usize) };
        let smem_a_at =
            |i: u32| unsafe { (&raw mut SMEM_A as *mut u8).add(i as usize * A_TILE_BYTES) };
        let smem_b_at =
            |i: u32| unsafe { (&raw mut SMEM_B as *mut u8).add(i as usize * B_TILE_BYTES) };

        // --- Stage 0: Initialize Barriers and TMEM ---
        let tile_bar = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut TILE_READY);
        let clc_bar = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut CLC_BAR);

        if thread0 {
            let mut i = 0u32;
            while i < STAGES {
                unsafe {
                    mbarrier_init(tma_bar_at(i), 1);
                    mbarrier_init(mma_bar_at(i), 1);
                }
                i += 1;
            }
            let mut i = 0u32;
            while i < NUM_ACCUM_STAGES {
                unsafe {
                    mbarrier_init(accum_full_at(i), 1);
                    mbarrier_init(accum_empty_at(i), 128); // number of epilogue threads
                }
                i += 1;
            }
        }
        // tile_bar.init runs the async-proxy fence + syncthreads, which also
        // publishes the raw inits above.
        let tile_bar = unsafe { tile_bar.init(1) };
        let clc_bar = unsafe { clc_bar.init(1) };

        let tmem = TmemGuard::<TmemUninit, { BN as u32 * NUM_ACCUM_STAGES }>::from_static(
            &raw mut TMEM_ADDR as *mut u32,
        );
        let tmem = unsafe { tmem.alloc() };
        thread::sync_threads();

        let mma_desc = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M128_N128)
            .element_type(Tcgen05ElementType::BF16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();

        // pre signal mma bars so they don't stall on the first iter: an
        // S-stage pipeline starts with S free-buffer credits.
        if thread0 {
            let mut i = 0u32;
            while i < STAGES {
                unsafe { mbarrier_arrive(mma_bar_at(i)) };
                i += 1;
            }
        }
        sync_threads();

        // --- Stage 1: K-Loop ---
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

                let (smem_a, smem_b) = (smem_a_at(phase), smem_b_at(phase));
                let (mma_bar, tma_bar) = (mma_bar_at(phase), tma_bar_at(phase));

                // Wait for MMA computation to finish, freeing the tile.
                unsafe {
                    while !mbarrier_try_wait_parity(mma_bar, parity) {}
                }

                if is_lane0 {
                    let k_offset = (k_idx as usize * BK) as i32;
                    unsafe {
                        cp_async_bulk_tensor_2d_g2s(smem_a, a, k_offset, m_offset, tma_bar);
                        cp_async_bulk_tensor_2d_g2s(smem_b, b, k_offset, n_offset, tma_bar);
                        mbarrier_arrive_expect_tx(tma_bar, 1, COMBINED_BYTES);
                    }
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

                        let (smem_a, smem_b) = (smem_a_at(phase), smem_b_at(phase));
                        let (mma_bar, tma_bar) = (mma_bar_at(phase), tma_bar_at(phase));

                        // Wait for MMA computation to finish, freeing the tile.
                        unsafe {
                            while !mbarrier_try_wait_parity(mma_bar, parity) {}
                        }

                        if is_lane0 {
                            let k_offset = (k_idx as usize * BK) as i32;
                            unsafe {
                                cp_async_bulk_tensor_2d_g2s(smem_a, a, k_offset, m_offset, tma_bar);
                                cp_async_bulk_tensor_2d_g2s(smem_b, b, k_offset, n_offset, tma_bar);
                                mbarrier_arrive_expect_tx(tma_bar, 1, COMBINED_BYTES);
                            }
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
                let (accum_empty, accum_full) =
                    (accum_empty_at(accum_stage), accum_full_at(accum_stage));

                if tile_iter >= NUM_ACCUM_STAGES {
                    let empty_parity = ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                    unsafe {
                        while !mbarrier_try_wait_parity(accum_empty, empty_parity) {}
                    }
                }

                let mut k_idx = 0u32;
                while k_idx < k_iters {
                    let phase = global_k % STAGES;
                    let parity = (global_k / STAGES) & 1;

                    let (smem_a, smem_b) = (smem_a_at(phase) as u64, smem_b_at(phase) as u64);
                    let (mma_bar, tma_bar) = (mma_bar_at(phase), tma_bar_at(phase));

                    // wait for tile memory to be filled
                    unsafe {
                        while !mbarrier_try_wait_parity(tma_bar, parity) {}
                    }

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
                            tcgen05_commit_shared_cluster(mma_bar as *mut u64);
                        }
                    }
                    k_idx += 1;
                    global_k += 1;
                }

                if is_lane0 {
                    unsafe {
                        tcgen05_commit_shared_cluster(accum_full as *mut u64);
                    }
                }

                tile_iter += 1;
            }
        }

        // --- Stage 2: TMEM -> Registers -> SMEM ---
        if warp_id < 4 {
            let mut epilogue_tile_iter = 0u32;
            let mut tile_parity = 0u32;
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

                let (full_bar, empty_bar) =
                    (accum_full_at(accum_stage), accum_empty_at(accum_stage));
                let tmem_stage_offset = accum_stage * BN as u32;

                let full_parity = (epilogue_tile_iter / NUM_ACCUM_STAGES) & 1;

                unsafe {
                    while !mbarrier_try_wait_parity(full_bar, full_parity) {}
                }

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

                unsafe {
                    mbarrier_arrive(empty_bar);
                }
                epilogue_tile_iter += 1;
            }
        }

        // --- Stage 4: Cleanup ---
        sync_threads();
        unsafe {
            // dealloc tmem
            let _dead = tmem.dealloc();
            // dealloc barriers
            if thread0 {
                let mut i = 0u32;
                while i < STAGES {
                    mbarrier_inval(tma_bar_at(i));
                    mbarrier_inval(mma_bar_at(i));
                    i += 1;
                }
                let mut i = 0u32;
                while i < NUM_ACCUM_STAGES {
                    mbarrier_inval(accum_full_at(i));
                    mbarrier_inval(accum_empty_at(i));
                    i += 1;
                }
            }
            let _tile_bar = tile_bar.inval();
            let _clc_bar = clc_bar.inval();
        }
    }

    // Per-stage-statics variant: the same kernel with every stage-indexed
    // array split into one static per stage and each selection done with a
    // `match` yielding (raw smem pointers, ManagedBarrier references) — the
    // gemm kernel's old idiom (pre "work stealing waves=2").
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub fn gemm_match(
        a: *const TmaDescriptor,
        b: *const TmaDescriptor, // reminder, B should be transposed to be NxK first
        mut c: DisjointSlice<u32>,
    ) {
        const BF16_SIZE: usize = 2;
        const A_TILE_BYTES: usize = BM * BK * BF16_SIZE;
        const B_TILE_BYTES: usize = BK * BN * BF16_SIZE;
        const COMBINED_BYTES: u32 = (A_TILE_BYTES + B_TILE_BYTES) as u32;
        const OUTPUT_SIZE: usize = BM * BN / BF16_SIZE;
        const CLUSTER_SIZE: u32 = 4;
        const TILES_M: u32 = (M / BM) as u32;
        const STAGES: u32 = 4;
        const NUM_ACCUM_STAGES: u32 = 2;

        // one static per stage — LLVM must select among symbol addresses
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

        static mut TMA_BAR_0: Barrier = Barrier::UNINIT;
        static mut TMA_BAR_1: Barrier = Barrier::UNINIT;
        static mut TMA_BAR_2: Barrier = Barrier::UNINIT;
        static mut TMA_BAR_3: Barrier = Barrier::UNINIT;
        static mut MMA_BAR_0: Barrier = Barrier::UNINIT;
        static mut MMA_BAR_1: Barrier = Barrier::UNINIT;
        static mut MMA_BAR_2: Barrier = Barrier::UNINIT;
        static mut MMA_BAR_3: Barrier = Barrier::UNINIT;

        static mut ACCUM_FULL_0: Barrier = Barrier::UNINIT;
        static mut ACCUM_FULL_1: Barrier = Barrier::UNINIT;
        static mut ACCUM_EMPTY_0: Barrier = Barrier::UNINIT;
        static mut ACCUM_EMPTY_1: Barrier = Barrier::UNINIT;
        static mut TILE_READY: Barrier = Barrier::UNINIT;

        static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;
        static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
        static mut CLC_BAR: Barrier = Barrier::UNINIT;

        const SBO_BYTES: u32 = 1024;
        const LBO_BYTES: u32 = 16;
        const TMA_WARP: u32 = 4;
        const MMA_WARP: u32 = 5;

        let tid = thread::threadIdx_x();
        let warp_id = warp::warp_id();
        let lane_id = tid % 32;
        let thread0 = tid == 0;

        // --- Stage 0: Initialize Barriers and TMEM ---
        let tma_bar_0 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR_0);
        let tma_bar_1 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR_1);
        let tma_bar_2 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR_2);
        let tma_bar_3 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR_3);
        let mma_bar_0 = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR_0);
        let mma_bar_1 = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR_1);
        let mma_bar_2 = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR_2);
        let mma_bar_3 = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR_3);
        let accum_full_0 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_FULL_0);
        let accum_full_1 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_FULL_1);
        let accum_empty_0 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_EMPTY_0);
        let accum_empty_1 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_EMPTY_1);
        let tile_bar = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut TILE_READY);
        let clc_bar = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut CLC_BAR);

        let tma_bar_0 = unsafe { tma_bar_0.init(1) };
        let tma_bar_1 = unsafe { tma_bar_1.init(1) };
        let tma_bar_2 = unsafe { tma_bar_2.init(1) };
        let tma_bar_3 = unsafe { tma_bar_3.init(1) };
        let mma_bar_0 = unsafe { mma_bar_0.init(1) };
        let mma_bar_1 = unsafe { mma_bar_1.init(1) };
        let mma_bar_2 = unsafe { mma_bar_2.init(1) };
        let mma_bar_3 = unsafe { mma_bar_3.init(1) };
        let accum_full_0 = unsafe { accum_full_0.init(1) };
        let accum_full_1 = unsafe { accum_full_1.init(1) };
        let accum_empty_0 = unsafe { accum_empty_0.init(128) }; // number of epilogue threads
        let accum_empty_1 = unsafe { accum_empty_1.init(128) };
        let tile_bar = unsafe { tile_bar.init(1) };
        let clc_bar = unsafe { clc_bar.init(1) };

        let tmem = TmemGuard::<TmemUninit, { BN as u32 * NUM_ACCUM_STAGES }>::from_static(
            &raw mut TMEM_ADDR as *mut u32,
        );
        let tmem = unsafe { tmem.alloc() };
        thread::sync_threads();

        let mma_desc = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M128_N128)
            .element_type(Tcgen05ElementType::BF16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();

        // pre signal mma bars so they don't stall on the first iter: an
        // S-stage pipeline starts with S free-buffer credits.
        if thread0 {
            mma_bar_0.arrive();
            mma_bar_1.arrive();
            mma_bar_2.arrive();
            mma_bar_3.arrive();
        }
        sync_threads();

        // --- Stage 1: K-Loop ---
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
                        #[unroll]
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
        if warp_id < 4 {
            let mut epilogue_tile_iter = 0u32;
            let mut tile_parity = 0u32;
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

                let (full_bar, empty_bar) = match accum_stage {
                    0 => (&accum_full_0, &accum_empty_0),
                    _ => (&accum_full_1, &accum_empty_1),
                };
                let tmem_stage_offset = accum_stage * BN as u32;

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
            let _ = tma_bar_0.inval();
            let _ = tma_bar_1.inval();
            let _ = tma_bar_2.inval();
            let _ = tma_bar_3.inval();
            let _ = mma_bar_0.inval();
            let _ = mma_bar_1.inval();
            let _ = mma_bar_2.inval();
            let _ = mma_bar_3.inval();
            let _ = accum_full_0.inval();
            let _ = accum_full_1.inval();
            let _ = accum_empty_0.inval();
            let _ = accum_empty_1.inval();
            let _tile_bar = tile_bar.inval();
            let _clc_bar = clc_bar.inval();
        }
    }

    // Branch-per-stage variant: same per-stage statics as `gemm_match`, but
    // each `match` duplicates the whole stage body into its arm instead of
    // yielding selected values. Inside an arm every resource is a
    // compile-time-known static, so nothing is ever dynamically selected —
    // the stage choice is a warp-uniform branch, paid for in code size.
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub fn gemm_branch(
        a: *const TmaDescriptor,
        b: *const TmaDescriptor, // reminder, B should be transposed to be NxK first
        mut c: DisjointSlice<u32>,
    ) {
        const BF16_SIZE: usize = 2;
        const A_TILE_BYTES: usize = BM * BK * BF16_SIZE;
        const B_TILE_BYTES: usize = BK * BN * BF16_SIZE;
        const COMBINED_BYTES: u32 = (A_TILE_BYTES + B_TILE_BYTES) as u32;
        const OUTPUT_SIZE: usize = BM * BN / BF16_SIZE;
        const CLUSTER_SIZE: u32 = 4;
        const TILES_M: u32 = (M / BM) as u32;
        const STAGES: u32 = 4;
        const NUM_ACCUM_STAGES: u32 = 2;

        // one static per stage, as in gemm_match
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

        static mut TMA_BAR_0: Barrier = Barrier::UNINIT;
        static mut TMA_BAR_1: Barrier = Barrier::UNINIT;
        static mut TMA_BAR_2: Barrier = Barrier::UNINIT;
        static mut TMA_BAR_3: Barrier = Barrier::UNINIT;
        static mut MMA_BAR_0: Barrier = Barrier::UNINIT;
        static mut MMA_BAR_1: Barrier = Barrier::UNINIT;
        static mut MMA_BAR_2: Barrier = Barrier::UNINIT;
        static mut MMA_BAR_3: Barrier = Barrier::UNINIT;

        static mut ACCUM_FULL_0: Barrier = Barrier::UNINIT;
        static mut ACCUM_FULL_1: Barrier = Barrier::UNINIT;
        static mut ACCUM_EMPTY_0: Barrier = Barrier::UNINIT;
        static mut ACCUM_EMPTY_1: Barrier = Barrier::UNINIT;
        static mut TILE_READY: Barrier = Barrier::UNINIT;

        static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;
        static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
        static mut CLC_BAR: Barrier = Barrier::UNINIT;

        const SBO_BYTES: u32 = 1024;
        const LBO_BYTES: u32 = 16;
        const TMA_WARP: u32 = 4;
        const MMA_WARP: u32 = 5;

        let tid = thread::threadIdx_x();
        let warp_id = warp::warp_id();
        let lane_id = tid % 32;
        let thread0 = tid == 0;

        // --- Stage 0: Initialize Barriers and TMEM ---
        let tma_bar_0 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR_0);
        let tma_bar_1 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR_1);
        let tma_bar_2 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR_2);
        let tma_bar_3 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut TMA_BAR_3);
        let mma_bar_0 = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR_0);
        let mma_bar_1 = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR_1);
        let mma_bar_2 = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR_2);
        let mma_bar_3 = ManagedBarrier::<Uninit, MmaBarrier>::from_static(&raw mut MMA_BAR_3);
        let accum_full_0 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_FULL_0);
        let accum_full_1 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_FULL_1);
        let accum_empty_0 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_EMPTY_0);
        let accum_empty_1 = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut ACCUM_EMPTY_1);
        let tile_bar = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut TILE_READY);
        let clc_bar = ManagedBarrier::<Uninit, Barrier>::from_static(&raw mut CLC_BAR);

        let tma_bar_0 = unsafe { tma_bar_0.init(1) };
        let tma_bar_1 = unsafe { tma_bar_1.init(1) };
        let tma_bar_2 = unsafe { tma_bar_2.init(1) };
        let tma_bar_3 = unsafe { tma_bar_3.init(1) };
        let mma_bar_0 = unsafe { mma_bar_0.init(1) };
        let mma_bar_1 = unsafe { mma_bar_1.init(1) };
        let mma_bar_2 = unsafe { mma_bar_2.init(1) };
        let mma_bar_3 = unsafe { mma_bar_3.init(1) };
        let accum_full_0 = unsafe { accum_full_0.init(1) };
        let accum_full_1 = unsafe { accum_full_1.init(1) };
        let accum_empty_0 = unsafe { accum_empty_0.init(128) }; // number of epilogue threads
        let accum_empty_1 = unsafe { accum_empty_1.init(128) };
        let tile_bar = unsafe { tile_bar.init(1) };
        let clc_bar = unsafe { clc_bar.init(1) };

        let tmem = TmemGuard::<TmemUninit, { BN as u32 * NUM_ACCUM_STAGES }>::from_static(
            &raw mut TMEM_ADDR as *mut u32,
        );
        let tmem = unsafe { tmem.alloc() };
        thread::sync_threads();

        let mma_desc = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M128_N128)
            .element_type(Tcgen05ElementType::BF16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();

        // pre signal mma bars so they don't stall on the first iter: an
        // S-stage pipeline starts with S free-buffer credits.
        if thread0 {
            mma_bar_0.arrive();
            mma_bar_1.arrive();
            mma_bar_2.arrive();
            mma_bar_3.arrive();
        }
        sync_threads();

        // --- Stage 1: K-Loop ---
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
                let k_offset = (k_idx as usize * BK) as i32;

                match phase {
                    0 => {
                        // Wait for MMA computation to finish, freeing the tile.
                        while !mma_bar_0.try_wait_parity(parity) {}
                        if is_lane0 {
                            unsafe {
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_A0 as *mut u8,
                                    a,
                                    k_offset,
                                    m_offset,
                                    tma_bar_0.as_ptr() as *mut Barrier,
                                );
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_B0 as *mut u8,
                                    b,
                                    k_offset,
                                    n_offset,
                                    tma_bar_0.as_ptr() as *mut Barrier,
                                );
                            }
                            tma_bar_0.arrive_expect_tx(COMBINED_BYTES);
                        }
                    }
                    1 => {
                        while !mma_bar_1.try_wait_parity(parity) {}
                        if is_lane0 {
                            unsafe {
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_A1 as *mut u8,
                                    a,
                                    k_offset,
                                    m_offset,
                                    tma_bar_1.as_ptr() as *mut Barrier,
                                );
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_B1 as *mut u8,
                                    b,
                                    k_offset,
                                    n_offset,
                                    tma_bar_1.as_ptr() as *mut Barrier,
                                );
                            }
                            tma_bar_1.arrive_expect_tx(COMBINED_BYTES);
                        }
                    }
                    2 => {
                        while !mma_bar_2.try_wait_parity(parity) {}
                        if is_lane0 {
                            unsafe {
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_A2 as *mut u8,
                                    a,
                                    k_offset,
                                    m_offset,
                                    tma_bar_2.as_ptr() as *mut Barrier,
                                );
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_B2 as *mut u8,
                                    b,
                                    k_offset,
                                    n_offset,
                                    tma_bar_2.as_ptr() as *mut Barrier,
                                );
                            }
                            tma_bar_2.arrive_expect_tx(COMBINED_BYTES);
                        }
                    }
                    _ => {
                        while !mma_bar_3.try_wait_parity(parity) {}
                        if is_lane0 {
                            unsafe {
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_A3 as *mut u8,
                                    a,
                                    k_offset,
                                    m_offset,
                                    tma_bar_3.as_ptr() as *mut Barrier,
                                );
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_B3 as *mut u8,
                                    b,
                                    k_offset,
                                    n_offset,
                                    tma_bar_3.as_ptr() as *mut Barrier,
                                );
                            }
                            tma_bar_3.arrive_expect_tx(COMBINED_BYTES);
                        }
                    }
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
                        let k_offset = (k_idx as usize * BK) as i32;

                        match phase {
                            0 => {
                                while !mma_bar_0.try_wait_parity(parity) {}
                                if is_lane0 {
                                    unsafe {
                                        cp_async_bulk_tensor_2d_g2s(
                                            &raw mut SMEM_A0 as *mut u8,
                                            a,
                                            k_offset,
                                            m_offset,
                                            tma_bar_0.as_ptr() as *mut Barrier,
                                        );
                                        cp_async_bulk_tensor_2d_g2s(
                                            &raw mut SMEM_B0 as *mut u8,
                                            b,
                                            k_offset,
                                            n_offset,
                                            tma_bar_0.as_ptr() as *mut Barrier,
                                        );
                                    }
                                    tma_bar_0.arrive_expect_tx(COMBINED_BYTES);
                                }
                            }
                            1 => {
                                while !mma_bar_1.try_wait_parity(parity) {}
                                if is_lane0 {
                                    unsafe {
                                        cp_async_bulk_tensor_2d_g2s(
                                            &raw mut SMEM_A1 as *mut u8,
                                            a,
                                            k_offset,
                                            m_offset,
                                            tma_bar_1.as_ptr() as *mut Barrier,
                                        );
                                        cp_async_bulk_tensor_2d_g2s(
                                            &raw mut SMEM_B1 as *mut u8,
                                            b,
                                            k_offset,
                                            n_offset,
                                            tma_bar_1.as_ptr() as *mut Barrier,
                                        );
                                    }
                                    tma_bar_1.arrive_expect_tx(COMBINED_BYTES);
                                }
                            }
                            2 => {
                                while !mma_bar_2.try_wait_parity(parity) {}
                                if is_lane0 {
                                    unsafe {
                                        cp_async_bulk_tensor_2d_g2s(
                                            &raw mut SMEM_A2 as *mut u8,
                                            a,
                                            k_offset,
                                            m_offset,
                                            tma_bar_2.as_ptr() as *mut Barrier,
                                        );
                                        cp_async_bulk_tensor_2d_g2s(
                                            &raw mut SMEM_B2 as *mut u8,
                                            b,
                                            k_offset,
                                            n_offset,
                                            tma_bar_2.as_ptr() as *mut Barrier,
                                        );
                                    }
                                    tma_bar_2.arrive_expect_tx(COMBINED_BYTES);
                                }
                            }
                            _ => {
                                while !mma_bar_3.try_wait_parity(parity) {}
                                if is_lane0 {
                                    unsafe {
                                        cp_async_bulk_tensor_2d_g2s(
                                            &raw mut SMEM_A3 as *mut u8,
                                            a,
                                            k_offset,
                                            m_offset,
                                            tma_bar_3.as_ptr() as *mut Barrier,
                                        );
                                        cp_async_bulk_tensor_2d_g2s(
                                            &raw mut SMEM_B3 as *mut u8,
                                            b,
                                            k_offset,
                                            n_offset,
                                            tma_bar_3.as_ptr() as *mut Barrier,
                                        );
                                    }
                                    tma_bar_3.arrive_expect_tx(COMBINED_BYTES);
                                }
                            }
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

                if tile_iter >= NUM_ACCUM_STAGES {
                    let empty_parity = ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                    match accum_stage {
                        0 => while !accum_empty_0.try_wait_parity(empty_parity) {},
                        _ => while !accum_empty_1.try_wait_parity(empty_parity) {},
                    }
                }

                let mut k_idx = 0u32;
                while k_idx < k_iters {
                    let phase = global_k % STAGES;
                    let parity = (global_k / STAGES) & 1;

                    match phase {
                        0 => {
                            // wait for tile memory to be filled
                            while !tma_bar_0.try_wait_parity(parity) {}
                            if is_lane0 {
                                let smem_a = &raw mut SMEM_A0 as u64;
                                let smem_b = &raw mut SMEM_B0 as u64;
                                let mut k_sub = 0;
                                #[unroll]
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
                                    tcgen05_commit_shared_cluster(mma_bar_0.as_ptr() as *mut u64);
                                }
                            }
                        }
                        1 => {
                            while !tma_bar_1.try_wait_parity(parity) {}
                            if is_lane0 {
                                let smem_a = &raw mut SMEM_A1 as u64;
                                let smem_b = &raw mut SMEM_B1 as u64;
                                let mut k_sub = 0;
                                #[unroll]
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
                                    tcgen05_commit_shared_cluster(mma_bar_1.as_ptr() as *mut u64);
                                }
                            }
                        }
                        2 => {
                            while !tma_bar_2.try_wait_parity(parity) {}
                            if is_lane0 {
                                let smem_a = &raw mut SMEM_A2 as u64;
                                let smem_b = &raw mut SMEM_B2 as u64;
                                let mut k_sub = 0;
                                #[unroll]
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
                                    tcgen05_commit_shared_cluster(mma_bar_2.as_ptr() as *mut u64);
                                }
                            }
                        }
                        _ => {
                            while !tma_bar_3.try_wait_parity(parity) {}
                            if is_lane0 {
                                let smem_a = &raw mut SMEM_A3 as u64;
                                let smem_b = &raw mut SMEM_B3 as u64;
                                let mut k_sub = 0;
                                #[unroll]
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
                                    tcgen05_commit_shared_cluster(mma_bar_3.as_ptr() as *mut u64);
                                }
                            }
                        }
                    }
                    k_idx += 1;
                    global_k += 1;
                }

                if is_lane0 {
                    unsafe {
                        match accum_stage {
                            0 => tcgen05_commit_shared_cluster(accum_full_0.as_ptr() as *mut u64),
                            _ => tcgen05_commit_shared_cluster(accum_full_1.as_ptr() as *mut u64),
                        }
                    }
                }

                tile_iter += 1;
            }
        }

        // --- Stage 2: TMEM -> Registers -> SMEM ---
        if warp_id < 4 {
            let mut epilogue_tile_iter = 0u32;
            let mut tile_parity = 0u32;
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
                let tmem_stage_offset = accum_stage * BN as u32;

                let full_parity = (epilogue_tile_iter / NUM_ACCUM_STAGES) & 1;

                match accum_stage {
                    0 => while !accum_full_0.try_wait_parity(full_parity) {},
                    _ => while !accum_full_1.try_wait_parity(full_parity) {},
                }

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

                match accum_stage {
                    0 => accum_empty_0.arrive(),
                    _ => accum_empty_1.arrive(),
                };
                epilogue_tile_iter += 1;
            }
        }

        // --- Stage 4: Cleanup ---
        sync_threads();
        unsafe {
            // dealloc tmem
            let _dead = tmem.dealloc();
            // dealloc barriers
            let _ = tma_bar_0.inval();
            let _ = tma_bar_1.inval();
            let _ = tma_bar_2.inval();
            let _ = tma_bar_3.inval();
            let _ = mma_bar_0.inval();
            let _ = mma_bar_1.inval();
            let _ = mma_bar_2.inval();
            let _ = mma_bar_3.inval();
            let _ = accum_full_0.inval();
            let _ = accum_full_1.inval();
            let _ = accum_empty_0.inval();
            let _ = accum_empty_1.inval();
            let _tile_bar = tile_bar.inval();
            let _clc_bar = clc_bar.inval();
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side driver, mirroring kernels/gemm: bf16 inputs through 128B-swizzled
// K-major TMA descriptors, C returned as M×N bf16 packed two-per-u32.
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

/// Linear grid for the CLC kernels; the generated host stub applies the
/// `#[cluster_launch(4, 1, 1)]` dims itself.
fn launch() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (((M / BM) * (N / BN)) as u32, 1, 1),
        // 6 warps: 0-3 epilogue, 4 TMA producer, 5 MMA consumer.
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Descriptors + output buffer built once so repeated launches time only the
/// kernel. Both variants launch through the same instance (same descriptors,
/// same C buffer) — the output cross-check reads C after each variant's runs.
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

    pub fn launch_arith(
        &mut self,
        stream: &CudaStream,
        module: &LoadedModule,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // SAFETY: launch() supplies the dims the kernel is written for (192
        // threads = 6 warps, linear grid, cluster 4 applied by the stub), and
        // the descriptors map buffers that outlive the launch.
        unsafe {
            module.gemm_arith(
                stream,
                launch(),
                self.dev_a_map.cu_deviceptr() as *const TmaDescriptor,
                self.dev_b_map.cu_deviceptr() as *const TmaDescriptor,
                &mut self.c,
            )?;
        }
        Ok(())
    }

    pub fn launch_match(
        &mut self,
        stream: &CudaStream,
        module: &LoadedModule,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // SAFETY: as in launch_arith.
        unsafe {
            module.gemm_match(
                stream,
                launch(),
                self.dev_a_map.cu_deviceptr() as *const TmaDescriptor,
                self.dev_b_map.cu_deviceptr() as *const TmaDescriptor,
                &mut self.c,
            )?;
        }
        Ok(())
    }

    pub fn launch_branch(
        &mut self,
        stream: &CudaStream,
        module: &LoadedModule,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // SAFETY: as in launch_arith.
        unsafe {
            module.gemm_branch(
                stream,
                launch(),
                self.dev_a_map.cu_deviceptr() as *const TmaDescriptor,
                self.dev_b_map.cu_deviceptr() as *const TmaDescriptor,
                &mut self.c,
            )?;
        }
        Ok(())
    }
}
