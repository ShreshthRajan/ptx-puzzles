// Puzzle 3: A triangular solve
//
// One step of a triangular solve. Four producer blocks each add their contribution into lsum with atomicAdd, then
// decrement a counter, fmod. The consumer block spins until fmod reaches 0, calls __threadfence(), and reads lsum.
//
// Make the consumer read the full sum. Par: 1.
//
// From: SuperLU_DIST's GPU triangular solve (xiaoyeli/superlu_dist#261, fixed upstream in 239280c).
#include "common.cuh"

constexpr int T = 128, P = 4;

__global__ void solve_step(int* fmod, float* lsum, float* x) {
  const int t = threadIdx.x;
  if (blockIdx.x > 0) {                           // producers 1..P
    atomicAdd(&lsum[t], (float)(blockIdx.x * t));
    __syncthreads();
    if (t == 0) {
      // >>> edit
      atomicSub(fmod, 1);
      __threadfence();
      // <<< edit
    }
  } else {                                        // consumer
    if (t == 0) { while (atomicAdd(fmod, 0) != 0) {} __threadfence(); }
    __syncthreads();
    x[t] = lsum[t] + 1.0f;
  }
}

int main() {
  const int deps = P;
  int* fmod = device_copy(&deps, 1);
  float* lsum = device_zeros<float>(T);
  float* x = device_zeros<float>(T);
  solve_step<<<1 + P, T>>>(fmod, lsum, x);
  finish_launch();
  float got[T], want[T];
  host_copy(got, x, T);
  for (int t = 0; t < T; ++t) want[t] = (float)(t * P * (P + 1) / 2) + 1.0f;
  return check("x", got, want, T);
}
