// nvcc-flags: -gencode arch=compute_100a,code=sm_100a -lcuda
// CUDA C++ port of the `gemm` kernel in ../src/lib.rs — same algorithm, written
// the way a CUDA programmer would write it, so the generated PTX shows what
// nvcc does differently from cuda-oxide's rustc codegen backend.
//
// Algorithm (identical to the Rust version):
//   C[8192x8192] = A[8192x8192] · B  (bf16 inputs, f32 TMEM accum, bf16 out)
//   - 128x128x64 tiles, 4-stage TMA->MMA smem pipeline
//   - 2-stage TMEM accumulator pipeline (tile N computes while N-1 drains)
//   - warp specialization: warps 0-3 epilogue, warp 4 TMA, warp 5 tcgen05 MMA
//   - CLC work stealing: each CTA does its own tile, then cancels not-yet-
//     launched clusters (4 CTAs) and processes their 4 tiles serially
//   - grid (4096,1,1), cluster (4,1,1), block (192,1,1)
//
// B must be pre-transposed to NxK so both TMA descriptors are K-major.

#include <cuda.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                       \
              cudaGetErrorString(err_));                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CU_CHECK(x)                                                            \
  do {                                                                         \
    CUresult err_ = (x);                                                       \
    if (err_ != CUDA_SUCCESS) {                                                \
      fprintf(stderr, "%s:%d: CUresult %d\n", __FILE__, __LINE__, (int)err_);  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

constexpr int M = 8192, N = 8192, K = 8192;
constexpr int BM = 128, BN = 128, BK = 64;
constexpr uint32_t STAGES = 4;        // TMA/MMA smem pipeline depth
constexpr uint32_t ACCUM_STAGES = 2;  // TMEM accumulator pipeline depth
constexpr uint32_t TILES_M = M / BM;
constexpr uint32_t TOTAL_TILES = (uint32_t)(M / BM) * (N / BN);
constexpr uint32_t CLUSTER_SIZE = 4;
constexpr uint32_t K_ITERS = K / BK;
constexpr int A_TILE_BYTES = BM * BK * 2;
constexpr int B_TILE_BYTES = BK * BN * 2;
constexpr uint32_t COMBINED_BYTES = A_TILE_BYTES + B_TILE_BYTES;
// smem-descriptor strides for the 128B-swizzled core-matrix layout
constexpr uint32_t LBO_BYTES = 16, SBO_BYTES = 1024;
// tcgen05 instruction descriptor: M128_N128, bf16 inputs, f32 accumulator
// (dtype F32 = 1<<4, atype/btype BF16 = 1<<7 | 1<<10, N/8 = 16<<17, M/16 = 8<<24)
constexpr uint32_t IDESC = 0x08200490u;

struct __align__(128) Smem {
  uint8_t a[STAGES][A_TILE_BYTES];
  uint8_t b[STAGES][B_TILE_BYTES];
  uint32_t out[BM * BN / 2];  // bf16 output tile packed two-per-u32
  uint32_t tmem_addr;
  uint32_t tile_info[3];  // tile_m, tile_n, has_work
  alignas(16) uint64_t clc_response[2];
  uint64_t tma_bar[STAGES];
  uint64_t mma_bar[STAGES];
  uint64_t accum_full[ACCUM_STAGES];
  uint64_t accum_empty[ACCUM_STAGES];
  uint64_t tile_ready;
  uint64_t clc_bar;
};

// ---------------------------------------------------------------------------
// PTX shims — CUDA C++ has no intrinsics for tcgen05/mbarrier-tx/TMA/CLC, so
// these mirror the instructions cuda-oxide emits (same semantics, and nvcc
// still owns all the surrounding codegen).

__device__ inline uint32_t smem_u32(const void *p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ inline void mbar_init(uint64_t *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(count));
}

__device__ inline void mbar_inval(uint64_t *bar) {
  asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(smem_u32(bar)));
}

__device__ inline void mbar_arrive(uint64_t *bar) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(smem_u32(bar))
               : "memory");
}

__device__ inline void mbar_arrive_expect_tx(uint64_t *bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(smem_u32(bar)),
      "r"(bytes)
      : "memory");
}

__device__ inline void wait_parity(uint64_t *bar, uint32_t parity) {
  uint32_t done;
  do {
    asm volatile(
        "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
        "selp.b32 %0, 1, 0, p;}"
        : "=r"(done)
        : "r"(smem_u32(bar)), "r"(parity)
        : "memory");
  } while (!done);
}

__device__ inline void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ inline void tma_load_2d(void *dst, const CUtensorMap *map,
                                   int32_t x, int32_t y, uint64_t *bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
      "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
      "l"(map), "r"(x), "r"(y), "r"(smem_u32(bar))
      : "memory");
}

__device__ inline void tmem_alloc(uint32_t *slot, uint32_t ncols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
          "r"(smem_u32(slot)),
      "r"(ncols)
      : "memory");
}

__device__ inline void tmem_dealloc(uint32_t addr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(
                   addr),
               "r"(ncols)
               : "memory");
}

// 64-bit shared-memory matrix descriptor: 16B-granular address/strides,
// bit 46 set, swizzle mode (128B = 2) in bits 61-63.
__device__ inline uint64_t smem_desc(uint32_t saddr) {
  return ((uint64_t)((saddr >> 4) & 0x3FFF)) |
         ((uint64_t)((LBO_BYTES >> 4) & 0x3FFF) << 16) |
         ((uint64_t)((SBO_BYTES >> 4) & 0x3FFF) << 32) | (1ull << 46) |
         (2ull << 61);
}

__device__ inline void mma_bf16(uint32_t d_tmem, uint64_t a_desc,
                                uint64_t b_desc, bool accumulate) {
  asm volatile(
      "{.reg .pred p; setp.ne.s32 p, %4, 0;\n"
      " .reg .u32 z; mov.u32 z, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {z, z, z, z}, "
      "p;}" ::"r"(d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(IDESC), "r"((uint32_t)accumulate)
      : "memory");
}

__device__ inline void mma_commit(uint64_t *bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
      "[%0];" ::"r"(smem_u32(bar))
      : "memory");
}

__device__ inline void tmem_ld_16x256b(uint32_t taddr, float &x, float &y,
                                       float &z, float &w) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
               : "r"(taddr));
}

__device__ inline void tmem_ld_wait() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// packs (lo, hi) -> (bf16(hi) << 16) | bf16(lo)
__device__ inline uint32_t cvt_bf16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(lo), "f"(hi));
  return d;
}

__device__ inline void stmatrix_x2(void *smem, uint32_t r0, uint32_t r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};" ::
                   "r"(smem_u32(smem)),
               "r"(r0), "r"(r1)
               : "memory");
}

__device__ inline void clc_try_cancel(uint64_t *resp, uint64_t *bar) {
  asm volatile(
      "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::"
      "complete_tx::bytes.b128 [%0], [%1];" ::"r"(smem_u32(resp)),
      "r"(smem_u32(bar))
      : "memory");
}

__device__ inline uint32_t clc_is_canceled(uint64_t lo, uint64_t hi) {
  uint32_t r;
  asm("{.reg .b128 rsp; mov.b128 rsp, {%1, %2};\n"
      " .reg .pred p; clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
      "p, rsp; selp.b32 %0, 1, 0, p;}"
      : "=r"(r)
      : "l"(lo), "l"(hi));
  return r;
}

__device__ inline uint32_t clc_first_ctaid_x(uint64_t lo, uint64_t hi) {
  uint32_t r;
  asm("{.reg .b128 rsp; mov.b128 rsp, {%1, %2};\n"
      " clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, "
      "rsp;}"
      : "=r"(r)
      : "l"(lo), "l"(hi));
  return r;
}

// ---------------------------------------------------------------------------
// Kernel

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(192)
    gemm_kernel(const __grid_constant__ CUtensorMap a_map,
                const __grid_constant__ CUtensorMap b_map, uint32_t *c) {
  extern __shared__ uint8_t smem_raw[];
  Smem &s = *reinterpret_cast<Smem *>(smem_raw);

  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid / 32;
  const uint32_t lane_id = tid % 32;

  // --- init barriers + TMEM ---
  if (tid == 0) {
    for (uint32_t i = 0; i < STAGES; i++) {
      mbar_init(&s.tma_bar[i], 1);
      mbar_init(&s.mma_bar[i], 1);
    }
    for (uint32_t i = 0; i < ACCUM_STAGES; i++) {
      mbar_init(&s.accum_full[i], 1);
      mbar_init(&s.accum_empty[i], 128);  // 4 epilogue warps arrive per thread
    }
    mbar_init(&s.tile_ready, 1);
    mbar_init(&s.clc_bar, 1);
    fence_proxy_async();  // make the inits visible to the TMA/MMA async proxy
  }
  __syncthreads();

  if (warp_id == 0) tmem_alloc(&s.tmem_addr, BN * ACCUM_STAGES);
  __syncthreads();
  const uint32_t tmem_base = s.tmem_addr;

  // pipeline starts with all stages free: give the producer STAGES credits
  if (tid == 0)
    for (uint32_t i = 0; i < STAGES; i++) mbar_arrive(&s.mma_bar[i]);
  __syncthreads();

  // --- warp 4: TMA producer + CLC work stealing ---
  if (warp_id == 4) {
    const bool lane0 = lane_id == 0;
    uint32_t global_k = 0;

    auto produce_tile = [&](uint32_t tile_m, uint32_t tile_n) {
      if (lane0) {
        s.tile_info[0] = tile_m;
        s.tile_info[1] = tile_n;
        s.tile_info[2] = 1;
        mbar_arrive(&s.tile_ready);
      }
      const int32_t m_off = (int32_t)(tile_m * BM);
      const int32_t n_off = (int32_t)(tile_n * BN);
      for (uint32_t k_idx = 0; k_idx < K_ITERS; k_idx++, global_k++) {
        const uint32_t stage = global_k % STAGES;
        const uint32_t parity = (global_k / STAGES) & 1;
        // wait for the MMA to release this stage's buffers
        wait_parity(&s.mma_bar[stage], parity);
        if (lane0) {
          const int32_t k_off = (int32_t)(k_idx * BK);
          tma_load_2d(s.a[stage], &a_map, k_off, m_off, &s.tma_bar[stage]);
          tma_load_2d(s.b[stage], &b_map, k_off, n_off, &s.tma_bar[stage]);
          mbar_arrive_expect_tx(&s.tma_bar[stage], COMBINED_BYTES);
        }
      }
    };

    // own tile first, then steal whole clusters until the queue is dry
    produce_tile(blockIdx.x % TILES_M, blockIdx.x / TILES_M);
    for (uint32_t clc_iter = 0;; clc_iter++) {
      uint32_t canceled = 0, first_stolen = 0;
      if (lane0) {
        fence_proxy_async();
        mbar_arrive_expect_tx(&s.clc_bar, 16);  // CLC response is 16 bytes
        clc_try_cancel(s.clc_response, &s.clc_bar);
        wait_parity(&s.clc_bar, clc_iter & 1);
        canceled = clc_is_canceled(s.clc_response[0], s.clc_response[1]);
        if (canceled)
          first_stolen = clc_first_ctaid_x(s.clc_response[0], s.clc_response[1]);
        fence_proxy_async();
      }
      canceled = __shfl_sync(0xffffffffu, canceled, 0);
      first_stolen = __shfl_sync(0xffffffffu, first_stolen, 0);

      if (!canceled) {
        if (lane0) {
          s.tile_info[2] = 0;
          mbar_arrive(&s.tile_ready);
        }
        break;
      }
      for (uint32_t step = 0; step < CLUSTER_SIZE; step++) {
        const uint32_t ctaid = first_stolen + step;
        produce_tile(ctaid % TILES_M, ctaid / TILES_M);
      }
    }
  }

  // --- warp 5: tcgen05 MMA consumer ---
  if (warp_id == 5) {
    const bool lane0 = lane_id == 0;
    uint32_t tile_parity = 0, global_k = 0;

    for (uint32_t tile_iter = 0;; tile_iter++) {
      wait_parity(&s.tile_ready, tile_parity);
      tile_parity ^= 1;
      if (s.tile_info[2] == 0) break;

      const uint32_t astage = tile_iter % ACCUM_STAGES;
      const uint32_t tmem_off = astage * BN;
      // wait for the epilogue to drain this accumulator (first uses are free)
      if (tile_iter >= ACCUM_STAGES)
        wait_parity(&s.accum_empty[astage],
                    ((tile_iter - ACCUM_STAGES) / ACCUM_STAGES) & 1);

      for (uint32_t k_idx = 0; k_idx < K_ITERS; k_idx++, global_k++) {
        const uint32_t stage = global_k % STAGES;
        const uint32_t parity = (global_k / STAGES) & 1;
        wait_parity(&s.tma_bar[stage], parity);
        if (lane0) {
          const uint32_t a_base = smem_u32(s.a[stage]);
          const uint32_t b_base = smem_u32(s.b[stage]);
#pragma unroll
          for (uint32_t k_sub = 0; k_sub < BK / 16; k_sub++) {
            // accumulate into D except on the tile's very first MMA
            mma_bf16(tmem_base + tmem_off, smem_desc(a_base + k_sub * 32),
                     smem_desc(b_base + k_sub * 32), k_idx > 0 || k_sub > 0);
          }
          mma_commit(&s.mma_bar[stage]);
        }
      }
      if (lane0) mma_commit(&s.accum_full[astage]);
    }
  }

  // --- warps 0-3: epilogue (TMEM -> regs -> smem -> global) ---
  if (warp_id < 4) {
    uint32_t tile_parity = 0;
    const uint32_t warp_row = warp_id * 32;
    // tcgen05.ld 16x256b returns 4 f32 per thread (16 rows x 8 cols); paired
    // loads at col+8 feed stmatrix.x2, whose two 8x8 tiles lanes 0-7 / 8-15
    // address (second matrix 16 bf16 columns to the right).
    const uint32_t row_within_8 = lane_id % 8;
    const uint32_t mat2_col = (lane_id >= 8 && lane_id < 16) ? 16 : 0;
    constexpr uint32_t ROW_STRIDE_BYTES = BN * 2;

    for (uint32_t tile_iter = 0;; tile_iter++) {
      wait_parity(&s.tile_ready, tile_parity);
      tile_parity ^= 1;
      if (s.tile_info[2] == 0) break;
      const uint32_t tile_m = s.tile_info[0];
      const uint32_t tile_n = s.tile_info[1];

      const uint32_t astage = tile_iter % ACCUM_STAGES;
      const uint32_t tmem_off = astage * BN;
      wait_parity(&s.accum_full[astage], (tile_iter / ACCUM_STAGES) & 1);

      uint8_t *out_bytes = reinterpret_cast<uint8_t *>(s.out);
#pragma unroll
      for (uint32_t row_off = 0; row_off < 32; row_off += 16) {
        const uint32_t tmem_row = warp_row + row_off;
#pragma unroll
        for (uint32_t col_block = 0; col_block < 8; col_block++) {
          const uint32_t col = col_block * 16;
          float ax, ay, az, aw, bx, by, bz, bw;
          tmem_ld_16x256b(tmem_base + tmem_off + (tmem_row << 16) + col,  //
                          ax, ay, az, aw);
          tmem_ld_wait();
          tmem_ld_16x256b(tmem_base + tmem_off + (tmem_row << 16) + col + 8,
                          bx, by, bz, bw);
          tmem_ld_wait();

          const uint32_t row_lo = tmem_row + row_within_8;
          stmatrix_x2(out_bytes + row_lo * ROW_STRIDE_BYTES + col * 2 + mat2_col,
                      cvt_bf16x2(ax, ay), cvt_bf16x2(bx, by));
          const uint32_t row_hi = row_lo + 8;  // staggered 8 rows down
          stmatrix_x2(out_bytes + row_hi * ROW_STRIDE_BYTES + col * 2 + mat2_col,
                      cvt_bf16x2(az, aw), cvt_bf16x2(bz, bw));
        }
      }

      // smem -> global, linear per-warp sweep for coalescing
      constexpr uint32_t TILE_W = BN / 2;  // packed-u32 columns per tile
      constexpr uint32_t WIDTH = N / 2;
      constexpr uint32_t PER_WARP = BM * TILE_W / 4;
      for (uint32_t idx = lane_id; idx < PER_WARP; idx += 32) {
        const uint32_t local_row = idx / TILE_W;
        const uint32_t local_col = idx % TILE_W;
        const uint32_t global_row = tile_m * BM + warp_row + local_row;
        const uint32_t global_col = tile_n * TILE_W + local_col;
        c[global_row * WIDTH + global_col] =
            s.out[(warp_row + local_row) * TILE_W + local_col];
      }

      mbar_arrive(&s.accum_empty[astage]);
    }
  }

  // --- cleanup ---
  __syncthreads();
  if (warp_id == 0) tmem_dealloc(tmem_base, BN * ACCUM_STAGES);
  __syncthreads();
  if (tid == 0) {
    for (uint32_t i = 0; i < STAGES; i++) {
      mbar_inval(&s.tma_bar[i]);
      mbar_inval(&s.mma_bar[i]);
    }
    for (uint32_t i = 0; i < ACCUM_STAGES; i++) {
      mbar_inval(&s.accum_full[i]);
      mbar_inval(&s.accum_empty[i]);
    }
    mbar_inval(&s.tile_ready);
    mbar_inval(&s.clc_bar);
  }
}

// ---------------------------------------------------------------------------
// Host

static CUtensorMap make_tensor_map(void *global, uint64_t width,
                                   uint64_t height, uint32_t tile_w,
                                   uint32_t tile_h) {
  CUtensorMap map;
  const uint64_t dims[2] = {width, height};
  const uint64_t strides[1] = {width * sizeof(__nv_bfloat16)};
  const uint32_t box[2] = {tile_w, tile_h};
  const uint32_t elem_strides[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, global, dims, strides, box,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

int main() {
  constexpr int WARMUP = 5, ITERS = 50;

  // bf16-rounded uniform [-1, 1) inputs; B transposed to NxK (K-major)
  std::vector<__nv_bfloat16> a_h((size_t)M * K), bt_h((size_t)N * K);
  {
    std::mt19937 rng_a(1), rng_b(2);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : a_h) v = __float2bfloat16(dist(rng_a));
    std::vector<float> b(  (size_t)K * N);
    for (auto &v : b) v = dist(rng_b);
    for (size_t n = 0; n < N; n++)
      for (size_t k = 0; k < K; k++)
        bt_h[n * K + k] = __float2bfloat16(b[k * N + n]);
  }

  __nv_bfloat16 *a_d, *bt_d;
  uint32_t *c_d;
  CUDA_CHECK(cudaMalloc(&a_d, a_h.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&bt_d, bt_h.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&c_d, (size_t)M * N / 2 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(a_d, a_h.data(), a_h.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(bt_d, bt_h.data(), bt_h.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(c_d, 0, (size_t)M * N / 2 * sizeof(uint32_t)));

  const CUtensorMap a_map = make_tensor_map(a_d, K, M, BK, BM);
  const CUtensorMap b_map = make_tensor_map(bt_d, K, N, BK, BN);

  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  sizeof(Smem)));
  auto launch = [&] {
    gemm_kernel<<<TOTAL_TILES, 192, sizeof(Smem)>>>(a_map, b_map, c_d);
  };

  for (int i = 0; i < WARMUP; i++) launch();
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < ITERS; i++) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float total_ms;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const double avg_ms = total_ms / ITERS;
  const double tflops =
      2.0 * M * (double)N * K / (avg_ms / 1e3) / 1e12;
  printf("gemm_baseline (CUDA C++)  %dx%dx%d  avg=%.4f ms  %.1f TFLOP/s\n", M,
         N, K, avg_ms, tflops);

  // spot-check 256 elements against a CPU reference on the same bf16 inputs
  std::vector<uint32_t> c_h((size_t)M * N / 2);
  CUDA_CHECK(cudaMemcpy(c_h.data(), c_d, c_h.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  std::mt19937 rng(42);
  double max_err = 0.0;
  for (int s = 0; s < 256; s++) {
    const size_t i = rng() % M, j = rng() % N;
    double truth = 0.0;
    for (size_t k = 0; k < K; k++)
      truth += (double)__bfloat162float(a_h[i * K + k]) *
               (double)__bfloat162float(bt_h[j * K + k]);
    const uint32_t packed = c_h[(i * N + j) / 2];
    const uint16_t bits = (j & 1) ? (uint16_t)(packed >> 16) : (uint16_t)packed;
    __nv_bfloat16 gb;
    memcpy(&gb, &bits, 2);
    max_err = fmax(max_err, fabs((double)__bfloat162float(gb) - truth));
  }
  // |C| ~ tens, bf16 out ulp ~ 0.25; anything past 1.0 is a real bug
  printf("%s spot check: max |err| = %.4f over 256 samples\n",
         max_err <= 1.0 ? "✓" : "✗", max_err);
  return max_err <= 1.0 ? 0 : 1;
}
