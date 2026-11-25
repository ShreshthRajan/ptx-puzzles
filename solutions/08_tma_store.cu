// Puzzle 8: The TMA store (sm_90 and newer)
//
// Each block's threads write a tile into shared memory, and the block syncs. Thread 0 then copies the tile to global
// memory with the TMA unit's bulk copy and waits for it to finish (bulk_store_and_wait in common.cuh).
//
// Make the copy read the finished tile. Par: 1.
//
// From: TMA store epilogues, such as CUTLASS's.
#include "common.cuh"

constexpr int B = 4, T = 128, TILE = 256;

__global__ void tile_store(float* g) {
  __shared__ alignas(128) float tile[TILE];
  const int t = threadIdx.x, b = blockIdx.x;
  for (int i = t; i < TILE; i += T) tile[i] = (float)(b * 1000 + i);
  // >>> edit
  fence_proxy_async();
  // <<< edit
  __syncthreads();
  if (t == 0) bulk_store_and_wait(g + b * TILE, tile, TILE * sizeof(float));
}

int main() {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  if (prop.major < 9) { printf("needs sm_90 or newer (this GPU is sm_%d%d)\n", prop.major, prop.minor); return 77; }
  float* g = device_zeros<float>(B * TILE);
  tile_store<<<B, T>>>(g);
  finish_launch();
  static float got[B * TILE], want[B * TILE];
  host_copy(got, g, B * TILE);
  for (int b = 0; b < B; ++b) for (int i = 0; i < TILE; ++i) want[b * TILE + i] = (float)(b * 1000 + i);
  return check("g", got, want, B * TILE);
}
