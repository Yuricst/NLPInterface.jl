# Intermittent GC corruption: investigation log

TODO.md item 1. Symptom: the test suite sometimes aborts with
`GC error (probable corruption)` (Julia's GC found an object whose type tag is
not a type while marking). Plan: forensics (step 1), control matrix (step 2),
targeted reduction (step 3), test-set bisection if needed (step 4),
acceptance runs (step 5), fix (step 6). One commit per step.

## Tools in this directory

- `Project.toml`: the package and its test dependencies in one environment, so
  `test/runtests.jl` runs in a single process (and under gdb) instead of
  through `Pkg.test`'s extra subprocess. `[sources]` needs Julia >= 1.11.
- `stress.sh LABEL K [julia flags]`: K parallel runs of the suite with
  `--check-bounds=yes --heap-size-hint=300M`, logs in `out/LABEL/`, classified
  by `classify.sh` as PASS, GCERR, CRASH or FAIL.
- `gdb_run.sh OUTFILE [julia flags]`: one run under gdb; on the abort it stops
  in `gc_dump_queue_and_abort` and prints the bad tag, the caller frames, and
  every thread's backtrace.

## Step 1: forensics

### Existing aborts (before this investigation)

| run | code | allocations | GC # | printed tag (`jl_(vt)`) | reported during |
|---|---|---|---|---|---|
| test_m2b | sparse branch | 198,323,822 | 78 | `Core.GotoIfNot(cond=SSAValue(73), dest=27)` | test_review_fixes.jl:7 |
| base_1 | d72207d | 102,508,653 | 147 | `<?#0x7ff67a6b8940::<circular reference @-1>>` | test_mock_snopt.jl:5 |
| base_3 | d72207d | 102,576,278 | 147 | `Base.length` | test_mock_snopt.jl:5 |
| base_4 | d72207d | 102,456,329 | 147 | `<?#0x7f04e68493d0::<circular reference @-1>>` | test_mock_snopt.jl:5 |

Reading:
- The "type" the GC found in the object's header is itself a valid Julia
  object (an IR node, a function) or a heap address. A header word that holds
  a pointer to another live object is the signature of a stale reference: the
  GC follows a pointer to memory that was freed and reused. A foreign buffer
  overrun would more likely leave raw doubles, integers or text there.
- The three d72207d aborts happened at the same GC (#147) after nearly the
  same number of allocations: the damage and its detection are close to
  deterministic, not random timing.
- "Reported during" is where the next full GC ran, not where the damage was
  done.

### Reproduction rate (Julia 1.12.1, heap hint 300M, `--check-bounds=yes`)

| code | harness | runs | GC error | segfault | pass |
|---|---|---|---|---|---|
| main (c3d3974) | direct `runtests.jl` | 15 | 0 | 0 | 15 |
| main (c3d3974) | under gdb | 6 | 0 | 0 | 6 |
| main (c3d3974) | `Pkg.test` | 5 | 0 | 0 | 5 |
| d72207d | direct `runtests.jl` | 5 | 3 | 1 | 1 |
| d72207d | `Pkg.test` | 5 | 3 | 0 | 2 |
| d72207d | under gdb | 18 | 10 | 2 | 6 |

- d72207d (before the sparse work) is a reliable reproducer (19/28, ~70%) with either
  harness. Every abort is detected at GC #146-148 while `test_mock_snopt.jl`
  builds the mock library in the parent process.
- main: 0/26. It was seen once on the sparse branch earlier, so the bug is most
  likely still there and only masked by the different allocation pattern (more
  tests run before the SNOPT tests). Use d72207d to investigate; check main
  again with the same instruments once the cause is known.

### Under gdb (`gdb_run.sh`, d72207d, 10 aborts caught at `gc_dump_queue_and_abort`)

- At detection, only thread 1 runs (in `gc_mark_loop_serial_`). The other
  Julia thread (worker thread 2, created for `Threads.@spawn` on 1.12) is idle in
  `ijl_task_get_next`, after `task_done_hook`; the rest are OpenBLAS and the
  signal thread. Nothing races with the GC at that moment.
- The bad object is always inside a 65,552 KiB mapping: Julia's GC pool
  pages, not a task stack and not a malloc'ed block.
- gdb disables ASLR, and the same object address recurs across independent
  runs: `0x7fffae57bf00` in 5 aborts, `0x7fffdf23fb30` in 2, `0x7fffae88dc40`
  in 1, `0x7fffae7128b0` in 1, `0x7fffae58e910` in 1. The corruption is
  deterministic in layout, so a hardware watchpoint on that header word is
  feasible (step 3).
- The "type tag" found there is pointer-like: either the object's own address
  (`vt == obj`, 4 aborts) or another address in the same pool region. Around
  `0x7fffdf23fb30` the memory is a chain of 16-byte cells holding pointers into
  the main thread's environment strings (`0x7ffffffeb...`) and to the next cell:
  data of the subprocess spawn running at that moment (`test_mock_snopt.jl`
  builds and runs the mock library). The cell was reused for unrelated data
  while a live object still pointed to it.

### Conclusion of step 1

The pattern is a stale reference (use-after-free) into a GC pool cell, not a
foreign buffer overrun: no raw doubles, integers or text in the header, and
the referenced cell holds well-formed data of a later allocation. The GC freed
an object that something still referenced, or something kept a pointer the GC
does not know about (a missing root or write barrier), and that something is
only reached by a later full GC. Which code holds the stale reference is not
visible from the abort (the parent is optimized out in the release build); the
control matrix (step 2) narrows the suspects, and the watchpoint (step 3)
finds the cell's history.

## Step 2: control matrix

`matrix.sh BASE_WORKTREE ENV110 K`: the reproducer (d72207d) with one factor
changed per cell, K = 8 runs per cell, two cells at a time.
`runners/subset.jl` runs a chosen subset of the test files (`NLPI_FILES`).

| cell | change from the reproducer | GC error | segfault | pass |
|---|---|---|---|---|
| A | none (Julia 1.12.1, real SNOPT, default threads) | 6 | 1 | 1 |
| B | SNOPT library not loaded (`SNOPT_LIB=`) | 0 | 0 | 8 |
| C | `--threads=1,0` (no worker thread) | 0 | 0 | 8 |
| D | Julia 1.12.3 | 7 | 0 | 1 |
| E | Julia 1.10.13 | 0 | 0 | 8 |
| F | without `test_snopt.jl` | 0 | 0 | 8 |
| G | without `test_problems.jl` and `test_ipopt.jl` (Ipopt) | 7 | 0 | 1 |

Reading:
- The corruption needs both the in-process SNOPT tests (B, F) and a second OS
  thread (C). Removing either one removes it, 0/8 against 7/8.
- Ipopt is not involved (G still fails; the abort only moves earlier, to GC #66,
  since less was allocated before).
- Not fixed in Julia 1.12.3 (D). Julia 1.10 (E) passes, but it is confounded:
  it has no interactive thread (so `Threads.@spawn` runs on the main thread,
  like C), and the `ScopedValue`-only nested-solve tests are skipped there.
- On Julia 1.12 the main task runs on thread 1 (interactive pool) and
  `Threads.@spawn` runs on thread 2. `test_snopt.jl` spawns at three places:
  line 8 (`wait(Threads.@spawn throw(e))`, outside any solve), line 311
  (`fetch(Threads.@spawn solve_snopt(...))` inside `fitness`, i.e. while
  thread 1 is inside SNOPT's callback) and line 322 (a worker task that runs a
  whole SNOPT solve on thread 2 after the outer solve). These are the
  suspects for step 3.

## Step 3: targeted reduction with stand-alone patterns

`repro/patterns.jl VARIANT N` loops one usage pattern from `test_snopt.jl`
with a full GC after every iteration; `repro/run_patterns.sh` runs variants
K times in parallel (`VARIANTS`, `SCRIPT` select what to run).

First pass (K = 4, N = 300, on main and on d72207d): no aborts at all for
`plain`, `spawn_outside`, `spawn_in_callback_trivial`,
`spawn_in_callback_nested`, `worker_solve` and `spawn_in_callback_ipopt`
(0/24 on each commit). None of the spawn sites on its own damages the heap, so
the trigger needs more of the "nested solve_snopt" testset (step 4).

Later variants (K = 6, N = 150, d72207d unless noted):

| variant | aborts |
|---|---|
| `nested_same_task` (fitness calls solve_snopt itself; it throws) | 0/6 |
| `worker_created_in_callback` (fitness spawns a task that solves after the outer solve) | 6/6, then 5/6 |
| `wcic_noop` (same task creation, later only allocates) | 0/6 |
| `wcic_ipopt` (same, later solves with Ipopt) | 0/6 |
| `wcic_from_ipopt` (task created in an Ipopt callback, later solves with SNOPT) | 0/6 |
| `scope_worker` (task created inside `_with_snopt_in_solve`, no callback, later solves with SNOPT) | 6/6; main: 6/6 |
| `scope_worker_with` (same, later `_with_snopt_in_solve` + allocation) | 0/6 |
| `scope_worker_ipopt` (same, later Ipopt inside `_with_snopt_in_solve`) | 6/6 |
| `scope_worker`, `--threads=1,0` | 0/6 |

So the callback is not needed and SNOPT is not needed: a task created inside
NLPInterface's `ScopedValue` scope (`_with_snopt_in_solve`), which later
enters such a scope again and runs a solve, is enough. It reproduces on main
too (6/6), so the bug is still present there; the full suite on main only
misses the detection window.
