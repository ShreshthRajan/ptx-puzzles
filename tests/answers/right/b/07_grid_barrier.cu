// Puzzle 7: A grid barrier
//
// A software grid barrier. At each barrier, every block's thread 0 reads the generation number, fences, and takes a
// ticket from a counter. The last block to arrive resets the counter to 0, fences, and increments the generation. The
// other blocks spin on the generation and fence. The kernel runs 1,024 steps with a barrier after each, because the bug
// only shows when two blocks arrive at almost the same moment.
//
// Make every step read the previous step's values. Par: 0.
//
// From: Lucebox's AtomicGridSync.
#include "common.cuh"

constexpr int B = 8, T = 128, N = B * T, STEPS = 1024;

__device__ void grid_sync(unsigned* count, unsigned* gen) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned g = *(volatile unsigned*)gen;
    __threadfence();
    if (atomicAdd(count, 1u) == gridDim.x - 1) {
      // >>> edit
      st_relaxed(count, 0);
      // <<< edit
      __threadfence();
      atomicAdd(gen, 1u);
    } else {
      while (*(volatile unsigned*)gen == g) {}
      __threadfence();
    }
  }
  __syncthreads();
}

__global__ void stencil(unsigned* a, unsigned* b, unsigned* count, unsigned* gen) {
  const int i = blockIdx.x * T + threadIdx.x;
  for (int s = 0; s < STEPS; ++s) {
    b[i] = a[(i + N - 1) % N] + 2 * a[i] + a[(i + 1) % N] + s;   // neighbours at a block's edge live in another block
    grid_sync(count, gen);
    unsigned* c = a; a = b; b = c;
  }
}

int main() {
  unsigned h[N], tmp[N];
  for (int i = 0; i < N; ++i) h[i] = i * 2654435761u;
  unsigned* a = device_copy(h, N);
  unsigned* b = device_zeros<unsigned>(N);
  unsigned* count = device_zeros<unsigned>(1);
  unsigned* gen = device_zeros<unsigned>(1);
  stencil<<<B, T>>>(a, b, count, gen);
  finish_launch();
  for (int s = 0; s < STEPS; ++s) {
    for (int i = 0; i < N; ++i) tmp[i] = h[(i + N - 1) % N] + 2 * h[i] + h[(i + 1) % N] + s;
    for (int i = 0; i < N; ++i) h[i] = tmp[i];
  }
  unsigned got[N];
  host_copy(got, STEPS % 2 ? b : a, N);
  return check("a", got, h, N);
}
