// Puzzle 7: The grid barrier's reset
//
// A software grid barrier. Each block's thread 0 reads the generation, fences, and takes a ticket from a counter. The
// last block to arrive resets the counter, fences, and bumps the generation. The others spin on the generation and
// fence. The kernel runs four stencil steps with a barrier after each, so the counter is reset and reused. The
// arrivals release and the waits acquire. The reset is a plain store, and the last arriver's ticket was a relaxed
// atomic, so the reset is not ordered after the other blocks' tickets. That is a write-write race on the counter. No
// load reads a stale value, so adversarial mode has nothing to change here; the race report is the evidence.
//
// Make the reset safe. Par: 0.
//
// Seen in: Lucebox's Qwen3.5 decode megakernel (AtomicGridSync::sync, `*counter = 0`). RESOLVE reported the same race.
#include "common.cuh"

constexpr int B = 8, T = 128, N = B * T, STEPS = 4;

__device__ void grid_sync(unsigned* count, unsigned* gen) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned g = *(volatile unsigned*)gen;
    __threadfence();
    if (atomicAdd(count, 1u) == gridDim.x - 1) {
      // >>> edit
      atomicExch(count, 0u);
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
