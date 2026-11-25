# PTX Puzzles

by [Shreshth Rajan](https://shreshthrajan.com)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShreshthRajan/ptx-puzzles/blob/main/puzzles.ipynb)

When a GPU kernel passes data between threads that don't share a barrier, it is easy to get the synchronization
almost right. The kernel gives the right answer on your GPU and passes its tests, and the PTX memory model still allows
it to give the wrong one. I made these twelve puzzles while learning the memory model. Each one is a small kernel with a
bug of this kind, and most come from real projects: LLM megakernels, a sparse linear solver, CUTLASS and one of CUDA's
own samples.

Your job is to make each kernel correct with as little synchronization as you can. A race detector,
[syncsan](https://github.com/ShreshthRajan/syncsan), grades the fix, so a kernel that only happens to work on your GPU
does not pass.

The easiest way to play is the Colab notebook above, on the free T4. Each puzzle is a cell you edit, with a grade cell
under it.

## What you need to know

Threads in one block can wait for each other with `__syncthreads()`. Threads in different blocks can't, so they talk
through global memory. One block writes its data and then sets a flag or increments a counter. The other block spins
until it sees the flag and then reads the data.

That is not enough on its own. The GPU can make the writes visible out of order, and the reading block can get old
values from its cache. The writer has to publish with a **release**, and the reader has to wait with an **acquire**.
Once the acquire reads the value the release wrote, the reader sees everything the writer wrote before the release. If
either half is missing, nothing is guaranteed.

Plain loads and stores are weak. Relaxed, volatile, acquire, release and atomic accesses are strong. Two accesses to
the same word race when one is a write, one is weak, and nothing orders them.

These are the tools. All of them are in [puzzles/common.cuh](puzzles/common.cuh).

| Tool | What it does |
|---|---|
| `st_relaxed`, `ld_relaxed`, a `volatile` access | Strong accesses that order nothing else |
| `st_release(p, v)` | A flag store that publishes everything this thread has seen |
| `ld_acquire(p)` | A flag load. Once it reads a published value, this thread also sees everything published with it |
| `atomicAdd`, `atomicSub`, `atomicCAS`, `atomicExch` | Indivisible updates. They are strong, and on their own they order nothing else |
| `__threadfence()`, `fence_acq_rel()` | Fences. Before a strong store or an atomic, a fence makes it a release. After a strong load or an atomic, a fence makes it an acquire |
| `__syncthreads()` | A barrier for the block. Everything any thread did before it comes before everything any thread does after it |
| `bar_sync(id, n)`, `bar_arrive(id, n)` | A barrier for n threads. A thread's writes before its `bar_arrive` come before the reads after the matching `bar_sync` |
| `fence_proxy_async()` | Needed in puzzle 8. The TMA unit reads shared memory by a different path than ordinary loads and stores, and this fence orders this thread's earlier shared-memory writes before it |

## Puzzle 1

Block 1 writes 32 words and sets a flag. Block 0 waits for the flag and copies the words. You can change only the lines
between the edit markers.

```cuda
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
```

On an H100 this copies the right words, and the grader fails it anyway:

```
Puzzle 1: Message passing  (puzzles/01_message_passing.cu)
  native       out: correct
  races        32 racing pairs; the first:
    ...
  adversarial  out: WRONG, 32 of 32 values differ; out[0] = 0, expected 1
  control      out: correct
  FAIL
```

## How you are graded

The grader runs your kernel four ways. Run normally, it must give the right output. Under syncsan, it must have no
races. In adversarial mode, syncsan gives every racy load the old value the memory model allows, and the output must
still be right. The control run swaps in the newest value instead of an old one, to check that the swapping itself
doesn't break the kernel.

A passing fix is scored by how many barriers, fences, acquires and releases you added. Par is the count in the
reference solution. Changing a plain access to an atomic, or changing what a loop waits for, costs nothing, so a par of
0 is possible. A fence on every line passes, but it is over par.

## The puzzles

| # | Puzzle | From |
|---|---|---|
| 1 | [Message passing](puzzles/01_message_passing.cu) | Hazy Research's Llama-1B megakernel ([Megakernels#16](https://github.com/HazyResearch/Megakernels/issues/16)) |
| 2 | [Waiting on a counter](puzzles/02_counter_wait.cu) | megakernel-zoo, Lucebox and AutoMegaKernel |
| 3 | [A triangular solve](puzzles/03_triangular_solve.cu) | SuperLU_DIST ([#261](https://github.com/xiaoyeli/superlu_dist/issues/261)) |
| 4 | [The last block](puzzles/04_last_block.cu) | CUDA's threadFenceReduction sample |
| 5 | [One thread signals for the block](puzzles/05_block_signal.cu) | the usual megakernel signal |
| 6 | [One thread waits for the block](puzzles/06_block_wait.cu) | the usual megakernel wait |
| 7 | [A grid barrier](puzzles/07_grid_barrier.cu) | Lucebox |
| 8 | [The TMA store](puzzles/08_tma_store.cu) | TMA store epilogues, such as CUTLASS's |
| 9 | [Argmax, row by row](puzzles/09_argmax.cu) | Mirage MPK ([mirage#843](https://github.com/mirage-project/mirage/issues/843)) |
| 10 | [Recycling a page](puzzles/10_page_recycle.cu) | Hazy Research's Llama-1B megakernel ([Megakernels#17](https://github.com/HazyResearch/Megakernels/issues/17)) |
| 11 | [Split-K, three ways](puzzles/11_split_k.cu) | CUTLASS Stream-K ([cutlass#2027](https://github.com/NVIDIA/cutlass/issues/2027)) |
| 12 | [The spinlock](puzzles/12_spinlock.cu) | the textbook GPU spinlock |

The answers, with a paragraph on why each one works, are in [solutions/](solutions/README.md).

## Running it on your own GPU

You need Linux on x86-64, an NVIDIA GPU from the T4 or RTX 20 series onward, and CUDA 12 with `nvcc` (12.8 for RTX 50
cards). Puzzle 8 needs an H100 or newer and is skipped on older GPUs.

```
pip install syncsan
syncsan build --arch 90     # once, for your GPU: 75 T4, 80 A100, 86 RTX 30, 89 L4 and RTX 40, 90 H100, 120 RTX 50
python grade.py             # every puzzle
python grade.py 3           # one puzzle
python grade.py 3 -v        # with every race report
```

With no GPU but a [Daytona](https://www.daytona.io) key in `DAYTONA_API_KEY`, `python remote/run.py -- 3` grades
puzzle 3 on a rented RTX 5090. [tests/](tests/) checks that every puzzle fails as given and every solution passes, and
its latest results are in [tests/results](tests/results/).

## Further reading

The rules come from the PTX manual's
[memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model), which
Lustig et al. formalize in
[A Formal Analysis of the NVIDIA PTX Memory Consistency Model](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model)
(ASPLOS 2019). syncsan's [how it works](https://github.com/ShreshthRajan/syncsan/blob/main/docs/how-it-works.md)
explains what the grader checks and what it can't see.

## Credit

The format comes from Sasha Rush's [GPU Puzzles](https://github.com/srush/GPU-Puzzles). The bugs come from the
projects above. SuperLU_DIST and CUTLASS have fixed theirs. MIT license.
