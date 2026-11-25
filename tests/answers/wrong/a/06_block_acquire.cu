// Puzzle 6: One thread waits for the block
//
// Block 1 publishes x correctly: every thread writes, the block syncs, thread 0 releases. In block 0, thread 0 spins and
// fences, which is an acquire. Warps 1 to 3 have their own work first (about 2 ms of it) and then read x. By then block
// 1 has long finished, so the reads look safe and on hardware they almost always are. Thread 0's acquire orders thread
// 0's later accesses. Warps 1 to 3 never synchronized with thread 0, so nothing orders their reads after block 1's
// writes.
//
// Make warps 1 to 3 read x after block 1 wrote it. Par: 1.
//
// Seen in: the usual megakernel wait, where one elected thread spins on the counter for its whole block.
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
    __syncwarp();
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
