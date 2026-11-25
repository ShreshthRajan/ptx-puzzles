# PTX Puzzles

by [Shreshth Rajan](https://shreshthrajan.com)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShreshthRajan/ptx-puzzles/blob/main/puzzles.ipynb)

These are twelve small CUDA kernels, and each one is missing a piece of synchronization. The bugs come from real
kernels: megakernels, a sparse solver, CUTLASS, and the reduction pattern in CUDA's own samples. Most of them give the
right answer on today's GPUs, which is why they shipped. The PTX memory model still allows them to give the wrong one.

You fix each kernel by adding the least synchronization that makes it correct. A race detector,
[syncsan](https://github.com/ShreshthRajan/syncsan), grades the fix, so a kernel that only happens to work on your GPU
does not pass.

The quickest way to start is the Colab notebook above, on the free T4. Each puzzle is a cell you edit, with a grade
cell under it.

## Puzzle 1

Block 1 writes 32 words and sets a flag. Block 0 waits for the flag and copies the words. You may change only the lines
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

On an H100 the kernel copies the right words. The grader still fails it:

```
Puzzle 1: Message passing  (puzzles/01_message_passing.cu)
  native       out: correct
  races        32 racing pairs; the first:
    syncsan: RACE (write then read) at 0x72b4d1e00200
    block 1 warp 0: STG.E ...  [message_passing(...) 01_message_passing.cu:18]
    block 0 warp 0: LDG.E ...  [message_passing(...) 01_message_passing.cu:26]
  adversarial  out: WRONG, 32 of 32 values differ; out[0] = 0, expected 1
  control      out: correct
  FAIL
```

A relaxed flag carries its own value and orders nothing else, so block 0 can see the flag set and still read the words
from before block 1 wrote them. syncsan finds the 32 unordered pairs of accesses, and its adversarial mode hands each
racy load the old value that the memory model allows, which turns the hidden bug into wrong output.

## How grading works

The grader compiles your file and runs it four times. The native run must give the right output. The run under syncsan
must report no races. The adversarial run, where every racy load returns the value from before the write, must still
give the right output. The control run sends the same loads through syncsan but hands them their current values, which
shows that the redirection alone does not change the result.

A passing fix also gets a cost: the number of barriers, fences and acquire or release operations in your edits, against
the reference solution's count, called par. Putting a fence everywhere passes, and it scores over par.

## The puzzles

| # | Puzzle | The bug | Seen in |
|---|---|---|---|
| 1 | [Message passing](puzzles/01_message_passing.cu) | a relaxed flag on both sides | Hazy Research's Llama-1B megakernel ([Megakernels#16](https://github.com/HazyResearch/Megakernels/issues/16)) |
| 2 | [The volatile spin](puzzles/02_volatile_spin.cu) | no fence after the consumer's spin | megakernel-zoo, Lucebox, AutoMegaKernel |
| 3 | [The missing release](puzzles/03_missing_release.cu) | the producer's counter decrement has no release | SuperLU_DIST ([#261](https://github.com/xiaoyeli/superlu_dist/issues/261), fixed in 239280c) |
| 4 | [The last block](puzzles/04_last_block.cu) | no fence before the ticket | CUDA's threadFenceReduction sample |
| 5 | [One thread signals for the block](puzzles/05_block_release.cu) | no barrier before thread 0's release | the usual megakernel signal |
| 6 | [One thread waits for the block](puzzles/06_block_acquire.cu) | no barrier after thread 0's acquire | the usual megakernel wait |
| 7 | [The grid barrier's reset](puzzles/07_grid_barrier.cu) | the counter reset is a plain store | Lucebox's `AtomicGridSync` |
| 8 | [The bulk store's proxy fence](puzzles/08_tma_store_fence.cu) | no `fence.proxy.async` before a TMA store | CUTLASS's `tma_store_fence` |
| 9 | [The reused reduction buffer](puzzles/09_smem_reuse.cu) | shared slots rewritten before warp 0 read them | Mirage MPK's argmax ([mirage#843](https://github.com/mirage-project/mirage/issues/843)) |
| 10 | [The page goes back too early](puzzles/10_page_release.cu) | a page freed before its last write | Hazy Research's Llama-1B megakernel ([Megakernels#17](https://github.com/HazyResearch/Megakernels/issues/17)) |
| 11 | [Split-K, three ways](puzzles/11_split_k.cu) | the final split waits for only one earlier split | CUTLASS Stream-K ([cutlass#2027](https://github.com/NVIDIA/cutlass/issues/2027)) |
| 12 | [The spinlock](puzzles/12_spinlock.cu) | a lock taken and freed with relaxed atomics | the textbook GPU spinlock |

Each file opens with the setup and the goal. [puzzles/common.cuh](puzzles/common.cuh) wraps the PTX operations you need
as one-line helpers, such as `ld_acquire`, `st_release`, `fence_acq_rel` and the proxy fence. The reference solutions,
with a paragraph on why each one works, are in [solutions/](solutions/README.md).

## Running it on your own GPU

You need Linux on x86-64, an NVIDIA GPU of compute capability 7.5 or newer (T4, RTX 20 series and later), and CUDA 12
with `nvcc`. RTX 50 cards need CUDA 12.8. Puzzle 8 uses the Hopper copy engine, so it needs an H100 or newer and is
skipped on older GPUs.

```
pip install syncsan
syncsan build --arch 90     # once, for your GPU: 75 T4, 80 A100, 86 RTX 30, 89 L4 and RTX 40, 90 H100, 120 RTX 50
python grade.py             # every puzzle
python grade.py 3           # one puzzle
python grade.py 3 -v        # with every race report
```

`make` and `make P=3` do the same. If you have no GPU but have a [Daytona](https://www.daytona.io) key in `DAYTONA_API_KEY`,
`python remote/run.py -- 3` grades puzzle 3 on a rented RTX 5090 and deletes the machine afterwards.

[tests/](tests/) holds the maintainers' checks: every puzzle fails as given, every solution passes, a set of wrong
answers fails and a set of other correct answers passes. The latest logs from an H100 and an RTX 5090 are in
[tests/results](tests/results/).

## Further reading

The rules behind every puzzle are in the PTX ISA's
[memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model). Lustig et
al., [A Formal Analysis of the NVIDIA PTX Memory Consistency Model](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model)
(ASPLOS 2019), formalizes it. syncsan's
[how it works](https://github.com/ShreshthRajan/syncsan/blob/main/docs/how-it-works.md) says what the grader checks
and what it cannot see.

## Credit

The format is Sasha Rush's [GPU Puzzles](https://github.com/srush/GPU-Puzzles). The bugs belong to the projects cited
above, whose authors confirmed or fixed several of them. MIT license.
