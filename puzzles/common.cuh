// This file holds the helpers the puzzles use. You don't edit it.
//
// Each device helper wraps the PTX instruction it is named after. The loads, stores, atomics and fences use .gpu scope,
// so they order accesses between thread blocks.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e_)); exit(2); } } while (0)

// ---- loads and stores ----
// A relaxed access is strong. Two strong accesses to the same word never race, and a relaxed access orders no other
// access.
__device__ __forceinline__ unsigned ld_relaxed(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_relaxed(unsigned* p, unsigned v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
// An acquire load orders this thread's later accesses after it. A release store orders this thread's earlier accesses,
// and whatever this thread has already synchronized with, before it.
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }

// ---- read-modify-writes ----
// atomicAdd, atomicSub, atomicCAS and atomicExch compile to relaxed atomics. These are atomicAdd with acquire, release
// or both.
__device__ __forceinline__ unsigned atom_add_release(unsigned* p, unsigned v) { unsigned r; asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
__device__ __forceinline__ unsigned atom_add_acquire(unsigned* p, unsigned v) { unsigned r; asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
__device__ __forceinline__ unsigned atom_add_acq_rel(unsigned* p, unsigned v) { unsigned r; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }

// ---- fences ----
// __threadfence() is fence.sc.gpu. fence_acq_rel() is the lighter fence.acq_rel.gpu. Either one turns a relaxed access
// after it into a release, and a relaxed access before it into an acquire.
__device__ __forceinline__ void fence_acq_rel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }

// ---- named barriers (bar.sync / bar.arrive) ----
// bar_sync(id, n) waits until n threads have reached barrier id with bar_sync or bar_arrive; bar_arrive(id, n) counts
// toward it without waiting. n is a multiple of 32. Barrier 0 is __syncthreads(). Writes before a bar_arrive are
// ordered before reads after the matching bar_sync.
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }

// ---- the async proxy (sm_90 and newer) ----
// The TMA unit reads shared memory through the async proxy. Ordinary loads and stores use the generic proxy, and
// ordering between threads does not carry across proxies. fence_proxy_async() orders this thread's earlier ordinary
// shared-memory writes before later async-proxy accesses.
__device__ __forceinline__ void fence_proxy_async() {
#if __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}
// Copy bytes from shared memory to global memory with the TMA unit's bulk copy, and wait until the copy has finished.
__device__ __forceinline__ void bulk_store_and_wait(void* dst, const void* src_smem, unsigned bytes) {
#if __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :: "l"(dst), "r"((unsigned)__cvta_generic_to_shared(src_smem)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
#endif
}

// ---- host side ----
// Compare n values with what they should be. Prints one line and returns the exit status the grader reads.
template <class T>
int check(const char* name, const T* got, const T* want, int n) {
  int bad = 0, first = -1;
  for (int i = 0; i < n; ++i) if (got[i] != want[i]) { if (first < 0) first = i; ++bad; }
  if (!bad) { printf("%s: correct\n", name); return 0; }
  printf("%s: WRONG, %d of %d values differ; %s[%d] = %g, expected %g\n", name, bad, n, name, first,
         (double)got[first], (double)want[first]);
  return 1;
}

template <class T>
T* device_zeros(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); CK(cudaMemset(p, 0, n * sizeof(T))); return p; }

template <class T>
T* device_copy(const T* h, size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); CK(cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice)); return p; }

template <class T>
void host_copy(T* h, const T* d, size_t n) { CK(cudaMemcpy(h, d, n * sizeof(T), cudaMemcpyDeviceToHost)); }

inline void finish_launch() { CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
