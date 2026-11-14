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

## Step 4: bisection, down to a Julia-only reproducer

`runners/snopt_testsets.jl` runs `test_snopt.jl` with only the `@testset`s
named in `NLPI_KEEP` (optionally `NLPI_REPEAT` times, after `NLPI_PRE_FILES`),
then forces full GCs with allocation in between. On d72207d, K = 4 or 8:

- all of `test_snopt.jl` alone: 8/8 aborts (all at GC #67);
- each testset alone: 0/4 for every one of the 8 testsets;
- leave one out: only removing "nested solve_snopt" stops it (0/4);
  every other removal still aborts (2/4 to 4/4);
- "nested solve_snopt" together with any other testset: aborts in 1/4 to 4/4;
  "nested solve_snopt" run three times: 2/4. The other testsets only add the
  allocation that gets the damage detected.

That testset's worker task (spawned in fitness, solving after the outer
solve) is the trigger (step 3). Removing NLPInterface step by step
(`repro/scoped.jl`, `repro/mwe.jl`, K = 6):

| scenario | default threads | `--threads=1,0` |
|---|---|---|
| task spawned in a `with` scope, re-enters `with`, raw Ipopt.jl solve | 6/6 | 0/6 |
| same with `qsort` and a full GC inside the Julia comparator | 6/6 | 6/6 |
| task spawned outside any scope (`plain_qsort_gc`) | 0/6 | |
| task spawned in the scope, does not re-enter `with` | 0/6 | |
| task re-enters `with` but did not inherit a scope | 0/6 | |
| task spawned in the scope, re-enters `with`, plain `GC.gc(true)` | 6/6 | |
| `mwe.jl`: `@spawn`/`@async`, Ref or Int values | 6/6 each | 6/6 each |
| `mwe.jl` `direct_ref`: nested `with` + GC in the same task | 0/6 | 0/6 |

`repro/mwe_min.jl`, no packages, aborts on the first GC on Julia 1.12.1 and
1.12.3, with and without a heap hint (4/4 each):

```julia
using Base.ScopedValues
const sv = ScopedValue(0)
for i in 1:200
    t = with(() -> @async(with(() -> GC.gc(true), sv => 2)), sv => 1)
    wait(t)
    GC.gc(true)
end
```

Controls (3 runs each): without the outer `with`, without the inner `with`, or
without both, it passes 3/3. The inner `with` may use a different
`ScopedValue`, or be Base's own `with_logger`: still 3/3 aborts.

### Conclusion of step 4

This is a bug in Julia 1.12's ScopedValues: a task created inside a `with`
scope (which inherits that scope), entering any `with` itself, with a full GC
inside, corrupts the heap. `with` is lowered to
`Expr(:tryfinally, body, nothing, scope)`, so the scope switch is done by
codegen and the runtime; it cannot be fixed in a package. No matching report
was found in the Julia issue tracker; JuliaLang/julia#63831 (a 1.12 GC crash
from a partially initialized Union box, fix #60924 not backported to 1.12) is
a possible root cause, since `Scope` objects hold `Union{Nothing, ...}`
fields, but that is unconfirmed.

How NLPInterface hits it: `solve_snopt` wraps the whole solve, including
`fitness`, in `with(_SNOPT_IN_SOLVE => active)` (Julia >= 1.11). Any task a
user spawns from `fitness` inherits that scope. If that task later enters any
`with` (another `solve_snopt`, `with_logger`, a library using ScopedValues)
and a GC runs inside it, the heap is corrupted. The test suite does exactly
this in "nested solve_snopt" (the worker task).

## Step 5: acceptance criteria and baseline

`acceptance.sh LABEL K CHECKOUT...` runs the package-level reproducers
(`scope_worker`, `worker_created_in_callback`, `nested_testset` from
`repro/patterns.jl`, N = 150 iterations each) K times against every checkout,
interleaved round by round in one batch.

Baseline, unfixed main (585573f), K = 10:

| reproducer | aborts |
|---|---|
| `scope_worker` | 10/10 |
| `worker_created_in_callback` | 9/10 |
| `nested_testset` | 7/10 |

Full-suite stress runs are not evidence on main (0/26 even with the bug, step
1), so they only serve as a regression check.

A fix is accepted when, in one interleaved batch with K = 10:
1. the fixed checkout has 0 aborts on all three reproducers (with the bug
   present, P(0/10) <= 0.3^10 = 6e-6 for the weakest one);
2. the unfixed checkout (the revert) still aborts, so the batch can detect it;
3. the mechanism is explained: the fix removes the condition shown in step 4
   (a scope that tasks spawned from `fitness` inherit), rather than changing
   timing or heap layout;
4. the full test suite passes with the latest and the minimum dependency
   versions, and the stressed suite on d72207d + fix shows 0 aborts in 10 runs
   (baseline there: 19/28).

The Julia bug itself (`repro/mwe_min.jl`) stays; it is reported upstream
separately.

## Step 6: fix

### Change (src/snopt.jl)

- `solve_snopt` no longer opens a `ScopedValue` scope. Same-task nesting is
  detected with task-local storage on every Julia version (what Julia 1.10
  already did). Tasks spawned from `fitness` inherit nothing, so the Julia bug
  of step 4 cannot be reached through NLPInterface, whatever those tasks do
  (another solve, `with_logger`, a library using ScopedValues).
- The protection the scope gave (a solve from a task spawned in `fitness`
  throws instead of deadlocking on `SNOPT_LOCK`) is replaced by a deadlock
  check: the lock records its owner task; a solve that cannot take the lock
  retries every 10 ms, and if the owner is waiting for the current task
  (`wait`/`fetch`/`@sync`, directly or through a chain of up to 8 waits, read
  from `Task.donenotify`'s wait queue) it throws the nested-solve error.
  Other waits (a worker fed through a `Channel`) still deadlock, with the
  existing 60 s warning. This now also works on Julia 1.10.

### Tests

- `test_snopt.jl` "nested solve_snopt": the spawned-task case runs on every
  Julia version; added a two-level `fetch` chain and `@sync`.
- `mock_snopt/runtests_mock.jl` "tasks spawned from fitness": the regression
  through the public API (a task spawned from `fitness` enters `with_logger`
  and runs a full GC after the solve, 30 times). Unfixed: aborts 3/3 (real
  SNOPT, scratch run); fixed: 3/3 pass. It runs in a subprocess, on CI.

### Acceptance (criteria of step 5)

1. Interleaved batch, K = 10, `acceptance.sh`:

   | reproducer | fixed (working tree) | unfixed (06691a5) |
   |---|---|---|
   | `scope_worker` | 0/10 | 10/10 |
   | `worker_created_in_callback` | 0/10 | 8/10 |
   | `nested_testset` | 0/10 | 8/10 |

2. The unfixed checkout in the same batch aborts 26/30 (the batch can detect it).
3. Mechanism: the fix removes the inherited scope, the condition shown
   necessary and sufficient in step 4; it does not change timing or layout.
4. Stressed full suite, one batch: d72207d + fix 0/10 aborts, unfixed d72207d
   4/5, fixed main 5/5 pass. Full suite on Julia 1.12.1 (latest versions)
   passes; on Julia 1.10.13 with the minimum versions 1040/1041 pass, the one
   failure being the known platform-dependent `rosenbrock_10d` Ipopt status
   (it fails the same way without this change).

The Julia bug itself (`repro/mwe_min.jl`) is not fixed by this; it should be
reported upstream (JuliaLang/julia), with the 7-line reproducer above.

## Upstream status (checked 2026-11-13)

Already known and fixed in Julia:

- JuliaLang/julia#59483 "GC error on Julia 1.12" (label `regression 1.12`):
  the corrupted child is the `scope` field of `Core.Task`. The explanation in
  JuliaLang/julia#60620 matches step 4: a task can inherit a scope from the
  task that spawned it; when the parent leaves that scope, nothing roots it
  any more, and the child task's later scope changes let the GC free it.
- Fixed by JuliaLang/julia#60647 ("Preserve the scope across the exception
  handler") and JuliaLang/julia#60617 ("add wb_back on all task switch
  paths"), merged 2026-01-14/15, backported in "Backports for 1.12.5"
  (JuliaLang/julia#60612). Julia 1.12.5 was released 2026-02-10.

Verified here with the official Julia 1.12.7 binary:

| check | Julia 1.12.1 | Julia 1.12.7 |
|---|---|---|
| `repro/mwe_min.jl` | 3/3 abort | 0/6 abort |
| public-API regression script on unfixed NLPInterface (c3d3974) | 2/2 abort | 0/4 abort |

So Julia 1.12.0 to 1.12.4 are affected; 1.12.5 and later are not. The change
in step 6 still matters for users on 1.12.0-1.12.4 (1.12.1 and 1.12.3 are
what this machine has), and it costs nothing on fixed versions. No new
upstream report is needed.
