// Puzzle 3: The missing release
//
// A triangular solve step. Four producer blocks each add a contribution into lsum with atomicAdd, then decrement a
// dependency counter, fmod. The consumer block spins until fmod reaches 0, fences, and reads lsum. The consumer side
// is a correct acquire. The producers decrement with a relaxed atomicSub and nothing before it, so their additions are
// not released: the consumer can see fmod at 0 and still read a partial sum.
//
// Make the consumer read the full sum. Par: 1.
//
// Seen in: SuperLU_DIST's GPU triangular solve. An earlier commit put back the consumer's fence after a port dropped
// it; the producer's release was still missing (xiaoyeli/superlu_dist#261, fixed upstream in 239280c).
#include "common.cuh"

constexpr int T = 128, P = 4;

__global__ void solve_step(int* fmod, float* lsum, float* x) {
  const int t = threadIdx.x;
  if (blockIdx.x > 0) {                           // producers 1..P
    atomicAdd(&lsum[t], (float)(blockIdx.x * t));
    __syncthreads();
    if (t == 0) {
      // >>> edit
      __threadfence();
      atomicSub(fmod, 1);
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
