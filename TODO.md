# TODO

State as of 2026-11-13 (branch `6-support-for-sparse-jacobian`).

## Done

- Code review of 2026-11-07: high-severity issues fixed in PR #1, medium in PR #2, low in PRs #3 and #4.
- Follow-up review fixes merged in PR #7 (`fix/review-bounds-keywords-functors`):
  - `solve_ipopt` defaults `bound_relax_factor` to `0.0`, so `fitness` is no longer evaluated outside `[x_lb, x_ub]`.
  - `:finitediff` in the solvers uses a bound-aware one-sided difference.
  - Plain `fitness` calls get a copy of the iterate.
  - SNOPT `Print_file`/`Summary_file`/`Major_print_level` keywords are matched ignoring case.
  - Callable structs are accepted as `fitness`.
- End-to-end problem tests (`test/test_problems.jl`) solve only the well-conditioned problems with `:finitediff`, so they pass under every platform's numerics. On failure they print a GitHub Actions `::warning`.
- Sparse Jacobian support (issue #6, branch `6-support-for-sparse-jacobian`): `jac_sparsity` keyword on both solvers (matrix pattern, `:auto` via SparseConnectivityTracer, or an ADTypes detector), column coloring with SparseMatrixColorings, `detect_jac_sparsity`, docs page `sparse_jac.md`, `benchmark/sparse_jacobian.jl`.

## Open: before registering in General

1. **Intermittent GC corruption in the test suite (pre-existing; blocker).** `Pkg.test(julia_args=["--heap-size-hint=300M"])` sometimes aborts with `GC error (probable corruption)`. On d72207d (before the sparse work) it did so in 3 of 4 runs, during `test_mock_snopt.jl`; on the sparse branch it was seen once in 7 runs, during `test_review_fixes.jl`. Every time, `test_snopt.jl` had already run the real SNOPT library in-process. The error surfaces at a later, unrelated garbage collection, so the place it is reported is not where the memory was damaged. First suspects: the in-process SNOPT path (`ccall`s with `Ptr`s into Julia arrays, `unsafe_wrap` of SNOPT's `x`/`F`/`G` in `usrfun`, the `iw`/`rw`/`cw` workspace and its `resize!` after `f_snmema`, the `@cfunction` trampoline and global `_SNOPT_*` refs). Not yet reproduced with the mock library alone. Next steps: reproduce with only `test_snopt.jl` in a loop with a small heap hint; bisect by test set; run under a Julia debug build or `--check-bounds=yes` with `GC.enable_logging`; try `valgrind`/ASan on the SNOPT calls.
2. **Return a result object instead of `(xopt, fopt, info)`.** Adding outputs to the 3-tuple after registering is a breaking change. Add a struct (e.g. `NLPResult`) with `xopt`, `fopt`, `status`, and the values already available but discarded: Lagrange multipliers (SNOPT `xmul`/`Fmul`, Ipopt's multipliers), constraint values at `xopt`, iteration and `fitness` evaluation counts, the raw solver status code. Keep `xopt, fopt, info = solve_...(...)` working by making it iterate as the old tuple.
3. **Release automation and hygiene.**
   - Add a TagBot workflow (`.github/workflows/TagBot.yml`), so registered versions get git tags and GitHub releases.
   - Add an Aqua.jl test (ambiguities, stale deps, compat bounds, piracy, unbound type parameters).
   - Decide on the package name: "NLPInterface" is generic and close to the NLPModels.jl ecosystem, which may draw comments during the new-package waiting period.

## Roadmap (non-breaking, after 0.1)

4. **User-supplied derivatives and other AD backends.** Let `deriv` accept an ADTypes backend (`AutoEnzyme()`, `AutoMooncake()`, ...) through DifferentiationInterface.jl, or a user function returning the Jacobian. ADTypes is already a dependency.
5. **Exact Hessian for Ipopt.** Ipopt is limited to L-BFGS (`hessian_approximation = "limited-memory"`); e.g. 174 iterations on `rosenbrock_10d` against SNOPT's 68. Offer an optional Hessian of the Lagrangian via ForwardDiff, sparse with the existing coloring machinery.
6. **Per-iteration user callback** for logging, progress plots, and early stopping (Ipopt's intermediate callback, SNOPT's status), instead of side effects in `fitness`, which also receives Dual and tracer inputs.
7. **Linear constraints and two-sided constraint bounds.** Pass a known linear part (SNOPT's `A` matrix; Ipopt `jac_c_constant`/`jac_d_constant`) and allow `l <= c(x) <= u` instead of only `ceq == 0`, `cineq <= 0`.
8. **Central finite differences** (`deriv = :central`): more accurate than the current one-sided steps; SNOPT with one-sided differences struggles at tight tolerances.
9. **Optimization.jl (SciML) wrapper**, to expose the SNOPT interface through a widely used API.

## Known limitations (cannot be removed in this package)

- `solve_snopt` called from a task that existed before the solve (e.g. a worker pool fed from `fitness`) still deadlocks on `SNOPT_LOCK`. This cannot be told apart from legitimate concurrent solves. A solve that waits 60 s (`SNOPT_LOCK_WAIT_WARN_SECONDS`) logs a warning, once per session, pointing it out. Nested solves from the calling task, or from tasks spawned in `fitness` on Julia ≥ 1.11 (ScopedValues), are rejected.
- With ForwardDiff, the values SNOPT sees during the solve come from the Dual evaluation. For `fitness` with adaptive inner solvers, they can differ from a plain call at the inner solver's tolerance. The returned `fopt` comes from a plain call at `xopt`, and the in-solve behaviour is documented.
- Exceptions rethrown after `IpoptSolve`/`f_snopta` lose their original backtrace, because Julia cannot rethrow with a stored one. The original backtrace is logged at debug level (`JULIA_DEBUG=NLPInterface`).
- The mock SNOPT library is built and run in CI on Linux and macOS only. On Windows it runs only with `NLPINTERFACE_REQUIRE_MOCK_SNOPT=1`, so DLL build and loading are untested there. The macOS relocation cache and case-insensitive file-name handling are exercised only by the macOS CI job.
- `IPOPT_REAL_OPTIONS` is the option list of Ipopt 3.14. A real-valued option added in a later Ipopt still needs a `Float64` value.
