#!/usr/bin/env bash
# Step 2: control matrix. Each cell is the reproducer (BASE, default d72207d)
# with one factor changed; K runs per cell, two cells at a time.
#
#   dev/gc_corruption/matrix.sh BASE_WORKTREE ENV110 [K]
#     BASE_WORKTREE  checkout of the reproducer commit, with
#                    dev/gc_corruption/Project.toml instantiated (Julia >= 1.11)
#     ENV110         environment with the reproducer dev'ed, for Julia 1.10
set -u
here="$(cd "$(dirname "$0")" && pwd)"
base="$1"; env110="$2"; k="${3:-8}"
jl1121=$(readlink -f "$(command -v julia)")            # 1.12.1
jl1123="$HOME/.juliaup/bin/julia +release"            # 1.12.3
jl110="$HOME/.juliaup/bin/julia +lts"                 # 1.10.13
runner="$here/runners/subset.jl"
common=(ROOT="$base" PROJECT="$base/dev/gc_corruption" TESTFILE="$runner" NLPI_TESTDIR="$base/test")
no_snopt_file="test_problems.jl,test_validation.jl,test_residuals.jl,test_ipopt.jl,test_mock_snopt.jl,test_review_fixes.jl"
no_ipopt_files="test_validation.jl,test_residuals.jl,test_snopt.jl,test_mock_snopt.jl,test_review_fixes.jl"

cell() {  # cell NAME ENV... -- JULIA_FLAGS...
    local name="$1"; shift
    local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    env "${common[@]}" "${envs[@]}" "$here/stress.sh" "m_$name" "$k" "$@" > "$here/out/m_$name.sum" 2>&1
    tail -1 "$here/out/m_$name.sum"
}

cell A_baseline       JULIA="$jl1121"                                   -- &
cell B_no_snopt_lib   JULIA="$jl1121" SNOPT_LIB= SNOPTDIR=              -- &
wait
cell C_threads_1_0    JULIA="$jl1121"                                   -- --threads=1,0 &
cell D_julia_1.12.3   JULIA="$jl1123"                                   -- &
wait
cell E_julia_1.10.13  JULIA="$jl110" PROJECT="$env110"                  -- &
cell F_no_snopt_tests JULIA="$jl1121" NLPI_FILES="$no_snopt_file"       -- &
wait
cell G_no_ipopt_tests JULIA="$jl1121" NLPI_FILES="$no_ipopt_files"      -- &
wait
