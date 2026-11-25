// Puzzle 12: The spinlock
//
// 64 blocks each fold one value into a shared record (a sum, a count and a maximum) under a lock. Thread 0 takes the
// lock with atomicCAS, updates the record with plain loads and stores, and frees the lock with atomicExch. The lock
// gives mutual exclusion on the lock word only. Both atomics are relaxed, so the record's accesses inside one block's
// critical section are not ordered with those in the next. A block can read the record as it was before the previous
// holder's update and lose that update.
//
// Make the lock protect the record. Par: 2.
//
// Seen in: the textbook GPU spinlock, and every kernel that copies it with atomicCAS and atomicExch alone.
#include "common.cuh"

constexpr int B = 64;

struct Record { unsigned sum, count, max; };

__global__ void locked_update(unsigned* lock, Record* rec) {
  if (threadIdx.x != 0) return;
  const unsigned v = blockIdx.x * 37 % B + 1;
  while (atomicCAS(lock, 0u, 1u) != 0u) {}
  // >>> edit
  // <<< edit
  rec->sum += v;
  rec->count += 1;
  if (v > rec->max) rec->max = v;
  // >>> edit
  __threadfence();
  // <<< edit
  atomicExch(lock, 0u);
}

int main() {
  unsigned* lock = device_zeros<unsigned>(1);
  Record* rec = device_zeros<Record>(1);
  locked_update<<<B, 32>>>(lock, rec);
  finish_launch();
  Record got;
  host_copy(&got, rec, 1);
  unsigned g[3] = {got.sum, got.count, got.max}, want[3] = {0, B, 0};
  for (int b = 0; b < B; ++b) { const unsigned v = b * 37 % B + 1; want[0] += v; if (v > want[2]) want[2] = v; }
  return check("record", g, want, 3);
}
