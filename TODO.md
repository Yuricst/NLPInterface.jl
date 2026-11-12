# TODO

State as of 2026-11-11 (`main` at ace86bf).

## Done

- Code review of 2026-11-07: high-severity issues fixed in PR #1, medium in PR #2, low in PRs #3 and #4.
- Follow-up review fixes merged in PR #7 (`fix/review-bounds-keywords-functors`):
  - `solve_ipopt` defaults `bound_relax_factor` to `0.0`, so `fitness` is no longer evaluated outside `[x_lb, x_ub]`.
  - `:finitediff` in the solvers uses a bound-aware one-sided difference.
  - Plain `fitness` calls get a copy of the iterate.
  - SNOPT `Print_file`/`Summary_file`/`Major_print_level` keywords are matched ignoring case.
  - Callable structs are accepted as `fitness`.
- End-to-end problem tests (`test/test_problems.jl`) solve only the well-conditioned problems with `:finitediff`, so they pass under every platform's numerics. On failure they print a GitHub Actions `::warning`.

## Open

- **Sparse Jacobian support** (issue #6, branch `6-support-for-sparse-jacobian`). The branch has no commits beyond `main` yet, and both solvers still pass a dense Jacobian.

## Known limitations (cannot be removed in this package)

- `solve_snopt` called from a task that existed before the solve (e.g. a worker pool fed from `fitness`) still deadlocks on `SNOPT_LOCK`. This cannot be told apart from legitimate concurrent solves. A solve that waits 60 s (`SNOPT_LOCK_WAIT_WARN_SECONDS`) logs a warning, once per session, pointing it out. Nested solves from the calling task, or from tasks spawned in `fitness` on Julia ≥ 1.11 (ScopedValues), are rejected.
- With ForwardDiff, the values SNOPT sees during the solve come from the Dual evaluation. For `fitness` with adaptive inner solvers, they can differ from a plain call at the inner solver's tolerance. The returned `fopt` comes from a plain call at `xopt`, and the in-solve behaviour is documented.
- Exceptions rethrown after `IpoptSolve`/`f_snopta` lose their original backtrace, because Julia cannot rethrow with a stored one. The original backtrace is logged at debug level (`JULIA_DEBUG=NLPInterface`).
- The mock SNOPT library is built and run in CI on Linux and macOS only. On Windows it runs only with `NLPINTERFACE_REQUIRE_MOCK_SNOPT=1`, so DLL build and loading are untested there. The macOS relocation cache and case-insensitive file-name handling are exercised only by the macOS CI job.
- `IPOPT_REAL_OPTIONS` is the option list of Ipopt 3.14. A real-valued option added in a later Ipopt still needs a `Float64` value.
