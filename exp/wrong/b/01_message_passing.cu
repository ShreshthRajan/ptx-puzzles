// Puzzle 1: Message passing
//
// Block 1 writes 32 words and then sets a flag. Block 0 waits for the flag and then copies the words. Both flag
// accesses are relaxed. A relaxed flag carries its own value and orders nothing else, so block 0 can see the flag set
// and still read words from before block 1 wrote them.
//
// Make block 0's copy read block 1's words. Par: 2.
//
// Seen in: Hazy Research's Llama-1B megakernel, which hands activations between SMs through a relaxed counter and a
// volatile spin with no fence on either side (HazyResearch/Megakernels#16).
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
    while (ld_acquire(flag) == 0) {}
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
