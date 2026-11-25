// Puzzle 11: Split-K, three ways
//
// Three blocks each compute one split of the K loop of the same output tile. The first split stores its partial in a
// workspace and releases a flag. The middle split waits for the flag, adds its partial into the workspace with
// atomicAdd, and moves on. The final split waits for the same flag, reads the workspace, adds its own partial, and
// writes the tile. Each wait is a correct acquire. The final split acquires what the first split released and nothing
// more, so the middle split's additions are not ordered before its reads.
//
// Make the final split read all three partials. Par: 2.
//
// Seen in: CUTLASS's SM90 Stream-K GEMM before 3.7.0, where the final split waited for one earlier split with
// wait_lt(..., 1) and read the workspace while the middle split's atomicAdd was still landing (NVIDIA/cutlass#2027).
#include "common.cuh"

constexpr int T = 256, SPLITS = 3, K = 12, KS = K / SPLITS;

__global__ void splitk_tile(const float* a, unsigned* flag, float* work, float* out) {
  const int s = blockIdx.x, t = threadIdx.x;
  float p = 0.0f;
  for (int k = s * KS; k < (s + 1) * KS; ++k) p += a[k * T + t];
  if (s == 0) {                                   // first split
    work[t] = p;
    __syncthreads();
    if (t == 0) { __threadfence(); atomicAdd(flag, 1u); }
  } else if (s == 1) {                            // middle split
    if (t == 0) { while (*(volatile unsigned*)flag < 1) {} __threadfence(); }
    __syncthreads();
    for (int i = 0; i < 10; ++i) __nanosleep(100000);   // its K range is longer: Stream-K splits are uneven
    atomicAdd(&work[t], p);
    // >>> edit
    __syncthreads();
    if (t == 0) { __threadfence(); atomicAdd(flag, 1u); }
    // <<< edit
  } else {                                        // final split
    if (t == 0) {
      // >>> edit
      while (*(volatile unsigned*)flag < 2) {}
      // <<< edit
      __threadfence();
    }
    __syncthreads();
    out[t] = work[t] + p;
  }
}

int main() {
  static float h[K * T];
  float want[T];
  for (int t = 0; t < T; ++t) {
    want[t] = 0.0f;
    for (int k = 0; k < K; ++k) { h[k * T + t] = (float)((k * 7 + t) % 13); want[t] += h[k * T + t]; }
  }
  float* a = device_copy(h, K * T);
  unsigned* flag = device_zeros<unsigned>(1);
  float* work = device_zeros<float>(T);
  float* out = device_zeros<float>(T);
  splitk_tile<<<SPLITS, T>>>(a, flag, work, out);
  finish_launch();
  float got[T];
  host_copy(got, out, T);
  return check("out", got, want, T);
}
