// Puzzle 9: Argmax, row by row
//
// One block finds the argmax of each of 16 rows. For each row, every warp finds the best of its 32 values, lane 0 of
// each warp writes its candidate into shared memory, the block syncs, and warp 0 picks the row's answer from the 8
// candidates. The loop then moves to the next row and reuses the same shared slots.
//
// Make every row's answer come from that row. Par: 1.
//
// From: Mirage MPK's argmax (mirage-project/mirage#843).
#include "common.cuh"

constexpr int T = 256, W = T / 32, ROWS = 16;

__device__ void warp_argmax(float& v, int& i) {
  for (int o = 16; o > 0; o >>= 1) {
    const float v2 = __shfl_down_sync(0xffffffffu, v, o);
    const int i2 = __shfl_down_sync(0xffffffffu, i, o);
    if (v2 > v) { v = v2; i = i2; }
  }
}

__global__ void argmax_rows(const float* x, int* out) {
  __shared__ float best_val[W];
  __shared__ int best_idx[W];
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  for (int r = 0; r < ROWS; ++r) {
    float v = x[r * T + t];
    int i = t;
    warp_argmax(v, i);
    if (lane == 0) { best_val[warp] = v; best_idx[warp] = i; }
    __syncthreads();
    if (warp == 0) {
      v = lane < W ? best_val[lane] : -1.0f;
      i = lane < W ? best_idx[lane] : -1;
      warp_argmax(v, i);
      if (lane == 0) out[r] = i;
    }
    // >>> edit
    __syncthreads();
    // <<< edit
  }
}

int main() {
  static float h[ROWS * T];
  int want[ROWS];
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < T; ++t) {
      h[r * T + t] = (float)((t * 37 + r * 11) % T);   // each row is a permutation of 0..255
      if ((t * 37 + r * 11) % T == T - 1) want[r] = t;
    }
  float* x = device_copy(h, ROWS * T);
  int* out = device_zeros<int>(ROWS);
  argmax_rows<<<1, T>>>(x, out);
  finish_launch();
  int got[ROWS];
  host_copy(got, out, ROWS);
  return check("out", got, want, ROWS);
}
