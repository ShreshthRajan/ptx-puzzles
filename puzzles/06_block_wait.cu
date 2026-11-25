// Puzzle 6: One thread waits for the block
//
// Block 1 writes x and signals correctly. In block 0, thread 0 spins on the counter and then fences. The other
// threads read x. Warps 1 to 3 first do about 2 ms of other work, and by then block 1 has long finished, so on real
// hardware their reads almost always see the new values.
//
// Make every thread in block 0 read x after block 1 wrote it. Par: 1.
//
// From: the usual megakernel wait, where one thread waits for its whole block.
#include "common.cuh"

constexpr int T = 128;

__device__ void other_work() { for (int i = 0; i < 20; ++i) __nanosleep(100000); }

__global__ void handoff(unsigned* ready, float* x, float* y) {
  const int t = threadIdx.x;
  if (blockIdx.x == 1) {                          // producer
    x[t] = 5.0f * t;
    __syncthreads();
    if (t == 0) { __threadfence(); atomicAdd(ready, 1u); }
  } else {                                        // consumer
    if (t == 0) { while (*(volatile unsigned*)ready == 0) {} __threadfence(); }
    else if (t >= 32) other_work();
    // >>> edit
    // <<< edit
    y[t] = x[t] + 2.0f;
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
  for (int t = 0; t < T; ++t) want[t] = 5.0f * t + 2.0f;
  return check("y", got, want, T);
}
