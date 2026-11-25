// Puzzle 2: Waiting on a counter
//
// Block 1's 128 threads each write one element of x, and the block syncs. Thread 0 then calls __threadfence() and
// increments a counter. In block 0, thread 0 spins on the counter with a volatile load, the block syncs, and every thread reads x.
//
// Make block 0's reads see block 1's writes. Par: 1.
//
// From: megakernel-zoo, Lucebox and AutoMegaKernel.
#include "common.cuh"

constexpr int T = 128;

__global__ void handoff(unsigned* ready, float* x, float* y) {
  const int t = threadIdx.x;
  if (blockIdx.x == 1) {                          // producer
    x[t] = 2.0f * t;
    __syncthreads();
    if (t == 0) { __threadfence(); atomicAdd(ready, 1u); }
  } else {                                        // consumer
    if (t == 0) {
      while (*(volatile unsigned*)ready == 0) {}
      // >>> edit
      __threadfence();
      // <<< edit
    }
    __syncthreads();
    y[t] = x[t] + 1.0f;
  }
}

int main() {
  unsigned* ready = device_zeros<unsigned>(1);
  float* x = device_zeros<float>(T);
  float* y = device_zeros<float>(T);
  handoff<<<2, T>>>(ready, x, y);
  finish_launch();
  float got[T], want[T];
  host_copy(got, y, T);
  for (int t = 0; t < T; ++t) want[t] = 2.0f * t + 1.0f;
  return check("y", got, want, T);
}
