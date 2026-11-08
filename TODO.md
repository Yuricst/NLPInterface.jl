# TODO

Open issues from the code review of 2026-11-07. The high-severity issues were fixed in PR #1.

## Medium

All fixed on branch `fix/medium-severity-review` (SNOPT exception status, nested `solve_snopt`, Ipopt Jacobian at every point, extra `fitness` call with ForwardDiff, `fopt = 0.0` on failure).

## Low

All addressed on branch `fix/low-severity-review`. What remains are limitations that cannot be removed in this package:

- `solve_snopt` called from a task that existed before the solve (e.g. a worker pool fed from `fitness`) still deadlocks on `SNOPT_LOCK`. This cannot be told apart from legitimate concurrent solves; a solve that waits 60 s logs a warning (once per session) pointing it out.
- With ForwardDiff, the values SNOPT sees during the solve come from the Dual evaluation (for `fitness` with adaptive inner solvers they can differ from a plain call at the inner solver's tolerance). The returned `fopt` is now from a plain call at `xopt`; the in-solve behaviour is documented.
- Exceptions rethrown after `IpoptSolve`/`f_snopta` lose their original backtrace (Julia cannot rethrow with a stored one); it is logged at debug level (`JULIA_DEBUG=NLPInterface`).
- The mock SNOPT library is built and run in CI on Linux and macOS only; on Windows it runs only with `NLPINTERFACE_REQUIRE_MOCK_SNOPT=1` (DLL build/loading untested). The macOS relocation cache and case-insensitive file-name handling are exercised only by the macOS CI job.
- `IPOPT_REAL_OPTIONS` is the option list of Ipopt 3.14; a real-valued option added in a later Ipopt still needs a `Float64` value.
