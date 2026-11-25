// Puzzle 4: The last block
//
// Each of 64 blocks sums its slice and thread 0 writes the block's partial. Thread 0 then takes a ticket from a
// counter. The block that draws the last ticket fences and adds up all 64 partials. The reader's side is a correct
// acquire. The writers take their tickets with a relaxed atomicAdd and nothing before it, so the last block can draw
// the last ticket and still read partials that were not yet written.
//
// Make the last block read every partial. Par: 1.
//
// Seen in: CUDA's threadFenceReduction sample and every reduction built on it (PyTorch's reduce_kernel among them). The
// fence before the ticket is the one that goes missing.
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
  if (t == 0) fence_acq_rel();                    // the reader's acquire: thread 0's ticket, then a fence
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
