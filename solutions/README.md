# Solutions

Spoilers. Each file here is the puzzle with the reference edit, and each paragraph below says why it works in the PTX
memory model's terms. The model's rule for all twelve: two accesses to one location race when at least one is a write,
they are not both strong (relaxed, acquire, release or atomic), and no happens-before edge orders them. Happens-before
between threads comes from a release that an acquire observes, from barriers, and from fences that turn a relaxed
access into one of those. A racing load may return any value the location held that the model does not rule out,
including the value from before the write.

**1. Message passing.** `st_release(flag, 1)` and `ld_acquire(flag)`. The acquire reads the value the release wrote, so
the release synchronizes with the acquire, and every write before the release in the producer's program order happens
before every read after the acquire. With both accesses relaxed there is no synchronizes-with edge: coherence orders the
flag's values and nothing else. A fence on each side (`__threadfence()` before the store, after the spin) also works at
the same cost; it is the form the next puzzles use.

**2. The volatile spin.** `__threadfence()` after the spin. A volatile load is `ld.volatile`, which the model treats as
relaxed. A relaxed load followed by a fence is an acquire pattern: once the load observes the producer's release, the
fence synchronizes with it, and the reads after the fence are ordered. `ld_acquire` in the spin is the other answer.
The block's `__syncthreads()` then carries thread 0's acquire to the other threads (puzzle 6).

**3. The missing release.** `__threadfence()` before `atomicSub`. A fence followed by a relaxed atomic is a release
pattern. The consumer's acquire pairs with a release, and the producer had none: its atomicSub was relaxed and the
fence was on the consumer's side only. Each producer's `__syncthreads()` before the fence makes the release cover its
other threads' additions. The fence is per producer, and the counter's atomic decrements form a release sequence, so
the consumer that reads 0 synchronizes with all four.

**4. The last block.** `__threadfence()` before the ticket. The writer's fence and relaxed ticket form a release; the
last block's ticket, thread 0's fence and the barrier after it form the acquire. A fence after the ticket does nothing
for the writer: a fence orders accesses before it with accesses after it, and the partial and the ticket are both
before it. CUDA's threadFenceReduction sample has the writer's fence and no reader fence. That version relies on
`fence.sc` being totally ordered: when the last block's own fence, which it runs before taking its ticket, comes after
every writer's fence in that order, the writers' partials are ordered before its reads. The model does not require that
order. The reader here fences with `fence.acq_rel`, which has no place in the SC order, so only a release before each
ticket orders the partials.

**5. One thread signals for the block.** `__syncthreads()` before thread 0's release. A release orders the writes that
happen before it, and a write by another warp happens before it only through synchronization, here the barrier. With
the barrier, warps 1 to 3's writes happen before thread 0's fence, and the release carries them.
`SuperLU_DIST`'s fix puts a `__syncwarp()` in the same place, where one lane decrements for its warp.

**6. One thread waits for the block.** `__syncthreads()` after the acquire. Thread 0's acquire orders thread 0's later
accesses. The barrier makes everything before it in thread 0 happen before everything after it in the other threads,
so their reads come after the acquire and therefore after block 1's writes. Waiting 2 ms is a timing argument; the model
has no notion of time, and adversarial mode returns the stale value however long the warps wait.

**7. The grid barrier's reset.** `atomicExch(count, 0u)`. Two strong accesses to one location never race: coherence
orders them, and it follows each thread's program order, so the reset comes after the last arriver's own ticket, which
came after every other ticket. The plain store was not strong, and nothing ordered it after the other blocks' tickets:
the last arriver's ticket is relaxed and its fence comes after the reset. Any strong store also works at no cost
(`st_relaxed(count, 0)`, or a volatile store, which PTX treats as relaxed). A `__threadfence()` between the ticket and
the reset removes the race too, by making the ticket an acquire, at a cost of one fence.

**8. The bulk store's proxy fence.** `fence_proxy_async()` in every thread before the barrier. The PTX model has more
than one proxy: ordinary loads and stores use the generic proxy, and the bulk-copy engine reads shared memory through
the async proxy. Happens-before between threads does not order accesses across proxies; a proxy fence does. Each thread
fences its own generic writes, the barrier orders those fences before thread 0, and thread 0 issues the copy. This is
CUTLASS's `tma_store_fence()` followed by the barrier.

**9. The reused reduction buffer.** `__syncthreads()` at the end of the loop. The barrier in the middle of the loop orders
this row's candidate writes before warp 0's reads. A second barrier orders warp 0's reads of row r before any warp's
writes for row r + 1. Two alternating buffers remove the race with no second barrier: a write for row r + 2 comes after
the barrier of row r + 1, which warp 0 reaches only after its reads of row r. That version needs edits outside the
markers, so it is not scored here.

**10. The page goes back too early.** `FREE_THREADS = 96` and `bar_arrive(FREE, FREE_THREADS)` in the consumer after it
writes L. A named barrier orders the writes before each arrival before the reads and writes after the waiting
`bar_sync`. The loader waited for the storer alone, so only the storer's accesses were ordered before its loads. With the
consumer arriving too, the consumer's write of L happens before the loader's writes into the same words.

**11. Split-K, three ways.** In the middle split, `__syncthreads()` and then thread 0's `__threadfence(); atomicAdd(flag,
1u)`; in the final split, wait for `flag >= 2`. The final split's acquire synchronizes with the release whose value it
reads. Reading 1 synchronizes with the first split only. Reading 2 synchronizes with the middle split's release, which
comes after the middle split's own acquire of the first split's release, so both partials are ordered before the final
split's reads. CUTLASS 3.7.0 fixed the Stream-K case by waiting for every earlier split.

**12. The spinlock.** `__threadfence()` after taking the lock and before freeing it. The fence after the successful
atomicCAS makes it an acquire; the fence before atomicExch makes the unlock a release. Each holder's unlock then
synchronizes with the next holder's lock, so the critical sections are ordered one after another. Without them the lock
word's atomics are ordered by coherence and the record's plain accesses by nothing, and the compiler may also move the
plain accesses past the relaxed atomics. The unlock's fence must come after the critical section: a release pattern
orders only what precedes its fence, so the fence after the lock does not also publish the update. `__threadfence_block()`
orders accesses only for threads of the same block, so it works on neither side. `atom.acquire` on the lock and
`st.release` on the unlock do the same job.
