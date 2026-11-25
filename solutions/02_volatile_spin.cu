// Puzzle 2: The volatile spin
//
// Block 1's threads each write one element of x. Thread 0 then fences and bumps a counter: that is a correct release.
// Block 0's thread 0 spins on the counter with a volatile load, the block syncs, and every thread reads x. A volatile
// load is a relaxed load. Nothing after the spin turns it into an acquire, so the reads of x are not ordered after
// block 1's writes.
//
// Make block 0's reads see block 1's writes. Par: 1.
//
// Seen in: megakernel-zoo's Qwen3-8B engine, Lucebox's Qwen3.5 decode kernel, and AutoMegaKernel (whose spin is
// atomicAdd(c, 0), commented as an acquire). All three released correctly and spun without a fence afterwards.
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
