// Puzzle 10: The page goes back too early
//
// A megakernel block runs instructions out of a shared-memory page, with one warp per role. For the first instruction,
// the consumer (warp 0) writes its output O into the page's first half and arrives on O_READY. The storer (warp 1)
// waits there, copies O to global memory, and frees the page by arriving on FREE. The loader (warp 2) waits on FREE
// and loads the second instruction's input into the page. After arriving on O_READY, the consumer also writes the
// first instruction's log-sum-exp L into the page's second half. The page is freed when its reader is done; its last
// writer was never asked. The consumer's late write of L can land on top of the second instruction's input.
//
// Make the loader wait for every write into the page. Par: 1.
//
// Seen in: Hazy Research's Llama-1B megakernel, where the attention storer finishes the QOL page without waiting on
// L_arrived and the next instruction TMA-loads into it (HazyResearch/Megakernels#17).
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
    const float l = a + 1.0f;            // L, computed after O went out
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
