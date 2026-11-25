// Puzzle 1: Message passing
//
// Block 1 writes 32 words, then sets a flag. Block 0 waits until it sees the flag, then copies the words.
//
// Make block 0 copy the words block 1 wrote. Par: 2.
//
// From: Hazy Research's Llama-1B megakernel (HazyResearch/Megakernels#16).
#include "common.cuh"

constexpr int N = 32;

__global__ void message_passing(unsigned* flag, unsigned* data, unsigned* out) {
  if (threadIdx.x != 0) return;
  if (blockIdx.x == 1) {                          // producer
    for (int i = 0; i < N; ++i) data[i] = i + 1;
    // >>> edit
    st_relaxed(flag, 1);
    // <<< edit
  } else {                                        // consumer
    // >>> edit
    while (ld_relaxed(flag) == 0) {}
    // <<< edit
    for (int i = 0; i < N; ++i) out[i] = data[i];
  }
}

int main() {
  unsigned* flag = device_zeros<unsigned>(1);
  unsigned* data = device_zeros<unsigned>(N);
  unsigned* out = device_zeros<unsigned>(N);
  message_passing<<<2, 32>>>(flag, data, out);
  finish_launch();
  unsigned got[N], want[N];
  host_copy(got, out, N);
  for (int i = 0; i < N; ++i) want[i] = i + 1;
  return check("out", got, want, N);
}
