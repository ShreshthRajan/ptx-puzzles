# PTX Puzzles

Twelve small CUDA kernels, each with one synchronization bug taken from a real kernel. You fix each one by adding the
least synchronization that makes it correct under the PTX memory model. A race detector grades you, so a fix that
happens to work on your GPU today does not pass.

The bugs are the ones persistent kernels and megakernels hit when thread blocks hand data to each other: a flag without
an acquire, a counter without a release, one thread signaling for a whole block, a grid barrier's reset, a TMA store
without its proxy fence, a shared-memory page handed back before its last write. Most of them were found in shipping
code, and some of those kernels gave wrong output because of them.

## How it grades

Each puzzle is one `.cu` file in `puzzles/`. You edit only between `// >>> edit` and `// <<< edit`. The grader compiles
the file and runs it four ways:

1. **native**: the output must be correct.
2. **under [syncsan](https://github.com/ShreshthRajan/syncsan)**: syncsan tracks happens-before as the PTX model defines
   it and reports every pair of accesses that nothing orders. There must be none.
3. **adversarial**: every load that races with a write returns the value from before that write, which the model allows.
   The output must still be correct.
4. **control**: the same loads redirected with their current values, to show the redirection alone changes nothing.

A pass also prints your cost: the barriers, fences and acquire/release operations inside your edits, against par, the
reference solution's count. Fencing everything passes and scores over par.

```
Puzzle 3: The missing release  (puzzles/03_missing_release.cu)
  native       x: correct
  races        1 racing pair:
    syncsan: RACE (write then read) at 0x7323d2600300
    block 4 warp 2: ATOMG.E.ADD.F32.FTZ.RN.STRONG.GPU PT, RZ, desc[UR4][R2.64], R5  [solve_step(int*, float*, float*)+0xa0 sm_20_atomic_functions.hpp:84 < 03_missing_release.cu:19]
    block 0 warp 2: LDG.E R4, desc[UR4][R4.64]  [solve_step(int*, float*, float*)+0x250 03_missing_release.cu:29]
  adversarial  x: WRONG, 127 of 128 values differ; x[1] = 1, expected 11
  control      x: correct
  FAIL
```

## Run it

The quickest way is Colab's free T4, with no install:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShreshthRajan/ptx-puzzles/blob/main/puzzles.ipynb).
Each puzzle is a cell you edit, with a grade cell under it.

To run it on your own machine, you need an NVIDIA GPU of compute capability 7.5 or newer (T4, RTX 20 series and later),
Linux on x86-64, and CUDA 12 with `nvcc` (12.8 or newer for RTX 50). Puzzle 8 needs sm_90 or newer and is skipped on
older GPUs.

```
pip install syncsan
syncsan build --arch 90            # once; your GPU's compute capability: 75 T4, 80 A100, 86 RTX 30, 89 L4 and RTX 40, 90 H100, 120 RTX 50
python grade.py                    # every puzzle
python grade.py 3                  # one puzzle
python grade.py 3 -v               # every race report and syncsan's summary
```

`make`, `make P=3` and `make solutions` do the same.

No GPU: `remote/run.py` rents one on [Daytona](https://www.daytona.io), uploads this folder with your edits, installs
syncsan, grades, and deletes the sandbox. An RTX 5090 costs about $2 an hour there, and grading all twelve takes about
three minutes, half of it setup.

```
pip install "daytona[v0-sdk]"
export DAYTONA_API_KEY=...
python remote/run.py               # all puzzles on an RTX 5090
python remote/run.py -- 3 7        # puzzles 3 and 7
```

## The puzzles

| # | Puzzle | The bug | Seen in |
|---|---|---|---|
| 1 | [Message passing](puzzles/01_message_passing.cu) | a relaxed flag on both sides | Hazy Research's Llama megakernel ([Megakernels#16](https://github.com/HazyResearch/Megakernels/issues/16)) |
| 2 | [The volatile spin](puzzles/02_volatile_spin.cu) | no fence after the consumer's spin | megakernel-zoo, Lucebox, AutoMegaKernel |
| 3 | [The missing release](puzzles/03_missing_release.cu) | the producer's counter decrement has no release | SuperLU_DIST ([#261](https://github.com/xiaoyeli/superlu_dist/issues/261), fixed in 239280c) |
| 4 | [The last block](puzzles/04_last_block.cu) | no fence before the ticket | CUDA's threadFenceReduction and its descendants |
| 5 | [One thread signals for the block](puzzles/05_block_release.cu) | no barrier before thread 0's release | the one-elected-thread signal |
| 6 | [One thread waits for the block](puzzles/06_block_acquire.cu) | no barrier after thread 0's acquire | the one-elected-thread wait |
| 7 | [The grid barrier's reset](puzzles/07_grid_barrier.cu) | the counter reset is a plain store | Lucebox's `AtomicGridSync` |
| 8 | [The bulk store's proxy fence](puzzles/08_tma_store_fence.cu) | no `fence.proxy.async` before a TMA store | CUTLASS's `tma_store_fence` |
| 9 | [The reused reduction buffer](puzzles/09_smem_reuse.cu) | shared slots rewritten before warp 0 read them | Mirage MPK's argmax ([mirage#843](https://github.com/mirage-project/mirage/issues/843)) |
| 10 | [The page goes back too early](puzzles/10_page_release.cu) | a page freed before its last write | Hazy Research's Llama megakernel ([Megakernels#17](https://github.com/HazyResearch/Megakernels/issues/17)) |
| 11 | [Split-K, three ways](puzzles/11_split_k.cu) | the final split waits for one earlier split | CUTLASS Stream-K ([cutlass#2027](https://github.com/NVIDIA/cutlass/issues/2027)) |
| 12 | [The spinlock](puzzles/12_spinlock.cu) | a lock taken and freed with relaxed atomics | the textbook GPU spinlock |

Each file opens with the setup and the goal. [puzzles/common.cuh](puzzles/common.cuh) has the PTX operations as
one-line helpers (`ld_acquire`, `st_release`, `fence_acq_rel`, named barriers, the proxy fence). The reference
solutions and a paragraph on each are in [solutions/](solutions/README.md).

## Reading

- The PTX ISA's [memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model),
  the source for every rule here.
- Lustig et al., [A Formal Analysis of the NVIDIA PTX Memory Consistency Model](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model) (ASPLOS 2019).
- syncsan's [how it works](https://github.com/ShreshthRajan/syncsan/blob/main/docs/how-it-works.md), for what the grader
  checks and what it cannot see.

## Credit

The format is Sasha Rush's [GPU Puzzles](https://github.com/srush/GPU-Puzzles) and its siblings. The bugs belong to the
projects cited above, whose authors confirmed or fixed several of them. MIT license.
