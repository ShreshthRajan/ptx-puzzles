// Puzzle 4: The last block
//
// Each of 64 blocks sums its slice of the input, and thread 0 writes the block's partial sum. Thread 0 then takes a
// ticket from a counter with atomicAdd. The block that draws the last ticket calls fence_acq_rel() and adds up all 64
// partials.
//
// Make the last block read every partial. Par: 1.
//
// From: CUDA's threadFenceReduction sample and the reductions built on it.
#include "common.cuh"

constexpr int B = 64, T = 128;

__device__ unsigned block_sum(unsigned* s) {      // sums s[0..T) in place; every thread gets the total
  for (int k = T / 2; k > 0; k >>= 1) { if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k]; __syncthreads(); }
  return s[0];
}

__global__ void sum_last_block(const unsigned* in, unsigned* partial, unsigned* tickets, unsigned* total) {
  __shared__ unsigned s[T];
  __shared__ bool last;
  const int t = threadIdx.x, b = blockIdx.x;
  s[t] = in[b * T + t];
  __syncthreads();
  const unsigned mine = block_sum(s);
  if (t == 0) {
    partial[b] = mine;
    // >>> edit
    __threadfence();
    last = atomicAdd(tickets, 1u) == gridDim.x - 1;
    // <<< edit
  }
  __syncthreads();
  if (!last) return;
  if (t == 0) fence_acq_rel();
  __syncthreads();
  s[t] = t < B ? partial[t] : 0;
  __syncthreads();
  const unsigned all = block_sum(s);
  if (t == 0) *total = all;
}

int main() {
  unsigned h[B * T], want = 0;
  for (int i = 0; i < B * T; ++i) { h[i] = i % 7 + 1; want += h[i]; }
  unsigned* in = device_copy(h, B * T);
  unsigned* partial = device_zeros<unsigned>(B);
  unsigned* tickets = device_zeros<unsigned>(1);
  unsigned* total = device_zeros<unsigned>(1);
  sum_last_block<<<B, T>>>(in, partial, tickets, total);
  finish_launch();
  unsigned got;
  host_copy(&got, total, 1);
  return check("total", &got, &want, 1);
}
