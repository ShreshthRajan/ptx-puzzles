// Puzzle 10: Recycling a page
//
// A megakernel block runs its instructions out of a page of shared memory, and each warp has one job. For the first
// instruction, the consumer (warp 0) writes its output O into the first half of the page and arrives on barrier O_READY.
// It then writes a second result, L, into the second half. The storer (warp 1) waits on O_READY, copies O to global
// memory, and frees the page by arriving on barrier FREE. The loader (warp 2) waits on FREE and loads the next
// instruction's input into the page.
//
// Make every output correct. Par: 1.
//
// From: Hazy Research's Llama-1B megakernel (HazyResearch/Megakernels#17).
#include "common.cuh"

constexpr int H = 32;                    // a page holds O[0..H) and then L[0..H)
constexpr int O_READY = 1, FREE = 2, LOADED = 3;
// >>> edit
constexpr int FREE_THREADS = 96;         // threads that meet at FREE: the consumer, the storer and the loader
// <<< edit

__global__ void megakernel_block(const float* in1, const float* in2, float* out_o, float* out_l, float* out2) {
  __shared__ float page[2 * H];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (warp == 0) {                       // consumer
    const float a = in1[lane];
    page[lane] = 2.0f * a;               // O
    bar_arrive(O_READY, 64);
    const float l = a + 1.0f;            // L
    page[H + lane] = l;
    out_l[lane] = l;
    // >>> edit
    bar_arrive(FREE, FREE_THREADS);
    // <<< edit
    bar_sync(LOADED, 64);                // second instruction
    out2[lane] = page[lane] * page[H + lane];
  } else if (warp == 1) {                // storer
    bar_sync(O_READY, 64);
    out_o[lane] = page[lane];
    bar_arrive(FREE, FREE_THREADS);
  } else {                               // loader
    bar_sync(FREE, FREE_THREADS);
    page[lane] = in2[lane];
    page[H + lane] = in2[H + lane];
    bar_arrive(LOADED, 64);
  }
}

int main() {
  float h1[H], h2[2 * H];
  for (int i = 0; i < H; ++i) h1[i] = (float)i;
  for (int i = 0; i < 2 * H; ++i) h2[i] = (float)(100 + i);
  float* in1 = device_copy(h1, H);
  float* in2 = device_copy(h2, 2 * H);
  float* out_o = device_zeros<float>(H);
  float* out_l = device_zeros<float>(H);
  float* out2 = device_zeros<float>(H);
  megakernel_block<<<1, 96>>>(in1, in2, out_o, out_l, out2);
  finish_launch();
  float got[H], want[H];
  int status = 0;
  host_copy(got, out_o, H); for (int i = 0; i < H; ++i) want[i] = 2.0f * h1[i];
  status |= check("out_o", got, want, H);
  host_copy(got, out2, H); for (int i = 0; i < H; ++i) want[i] = h2[i] * h2[H + i];
  status |= check("out2", got, want, H);
  return status;
}
