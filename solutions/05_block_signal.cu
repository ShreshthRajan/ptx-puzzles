// Puzzle 5: One thread signals for the block
//
// Block 1's 128 threads each write one element of x. Thread 0 then calls __threadfence() and increments a
// counter. Block 0 waits on the counter with an acquire and reads x.
//
// Make every element of x reach block 0. Par: 1.
//
// From: the usual megakernel signal, where one thread bumps the counter for its whole block.
#include "common.cuh"

constexpr int T = 128;

__global__ void handoff(unsigned* ready, float* x, float* y) {
  const int t = threadIdx.x;
  if (blockIdx.x == 1) {                          // producer
    x[t] = 3.0f * t;
    // >>> edit
    __syncthreads();
    // <<< edit
    if (t == 0) { __threadfence(); atomicAdd(ready, 1u); }
  } else {                                        // consumer
    if (t == 0) { while (*(volatile unsigned*)ready == 0) {} __threadfence(); }
    __syncthreads();
    y[t] = x[t] - 1.0f;
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
  for (int t = 0; t < T; ++t) want[t] = 3.0f * t - 1.0f;
  return check("y", got, want, T);
}
