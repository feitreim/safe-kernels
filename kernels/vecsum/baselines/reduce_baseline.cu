// Reference baseline: sum-reduce N f32s with NVIDIA's own libraries, the
// "strong opponent" to measure vecsum against.
//
// Two measurements, so the benchmarking pitfall is visible rather than hidden:
//
//   1. thrust::reduce  — the naive call. It re-allocates CUB's scratch buffer
//      (cudaMalloc + a *synchronizing* cudaFree) on EVERY call, so the timed
//      loop measures allocator churn, not the reduction. Looks ~10x too slow.
//
//   2. cub::DeviceReduce::Sum — the same underlying kernel, but with the temp
//      buffer allocated ONCE up front and the result left on the device. This
//      is the real, memory-bandwidth-bound throughput and the number to beat.
//
// Methodology mirrors kernels/vecsum/src/bin/bench.rs: same N, same
// WARMUP/ITERS, CUDA-event (device-side) timing, output kept on-device inside
// the timed loop (bench.rs likewise times only the launches, not `to_host_vec`).
//
//   nvcc -O3 -arch=native -o reduce_baseline reduce_baseline.cu && ./reduce_baseline

#include <cstdio>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

constexpr int N = 1 << 24;  // 16M elements — matches bench.rs
constexpr int WARMUP = 5;
constexpr int ITERS = 50;

// A reduction reads all N inputs once; the scalar output is negligible.
static double throughput_gbps(double avg_ms) {
    double gb = static_cast<double>(N) * sizeof(float) / 1.0e9;
    return gb / (avg_ms / 1.0e3);
}

int main() {
    // All ones, so the sum is exactly N — a trivial correctness check. Bandwidth
    // is independent of the data values, so this stands in for the pseudo-random
    // input the Rust harness uses.
    thrust::device_vector<float> a(N, 1.0f);
    float* d_in = thrust::raw_pointer_cast(a.data());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0.0f;

    // --- 1. Naive thrust::reduce (re-allocates scratch every call) ------------
    float thrust_result = 0.0f;
    for (int i = 0; i < WARMUP; ++i)
        thrust_result = thrust::reduce(a.begin(), a.end(), 0.0f);

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i)
        thrust_result = thrust::reduce(a.begin(), a.end(), 0.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_ms, start, stop);
    double thrust_ms = total_ms / ITERS;

    // --- 2. cub::DeviceReduce with temp storage hoisted out of the loop -------
    float* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);  // query size only
    cudaMalloc(&d_temp, temp_bytes);

    for (int i = 0; i < WARMUP; ++i)
        cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i)
        cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_ms, start, stop);
    double cub_ms = total_ms / ITERS;

    float cub_result = 0.0f;
    cudaMemcpy(&cub_result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("thrust::reduce     N=%d  avg=%.4f ms  throughput=%.1f GB/s  (naive: re-mallocs scratch)\n",
           N, thrust_ms, throughput_gbps(thrust_ms));
    printf("cub::DeviceReduce  N=%d  avg=%.4f ms  throughput=%.1f GB/s  (hoisted scratch <- beat this)\n",
           N, cub_ms, throughput_gbps(cub_ms));
    printf("✓ results (thrust=%.1f, cub=%.1f, expected %d)\n",
           thrust_result, cub_result, N);

    cudaFree(d_temp);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
