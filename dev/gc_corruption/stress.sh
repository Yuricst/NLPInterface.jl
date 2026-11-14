#!/usr/bin/env bash
# Run test/runtests.jl in K parallel processes and classify each run.
#
#   dev/gc_corruption/stress.sh LABEL K [extra julia flags...]
#
# Environment knobs (all optional):
#   JULIA          julia command (default: julia), e.g. "$HOME/.juliaup/bin/julia +release"
#   PROJECT        project to run in (default: this directory)
#   HEAP_HINT      --heap-size-hint value (default: 300M)
#   OUT            output directory (default: dev/gc_corruption/out/LABEL)
#   TESTFILE       script to run (default: test/runtests.jl)
#   PKGTEST=1      run through `Pkg.test(julia_args=...)` in ROOT instead (the
#                  way the original aborts were seen); PROJECT/TESTFILE unused
#   ROOT           package directory (default: the repository holding this script)
# Any other variables (SNOPT_LIB, JULIA_NUM_THREADS, ...) are passed through.
#
# Prints one line per run: PASS, GCERR (GC error (probable corruption)), CRASH
# (other signal), or FAIL (test failures), then a summary line.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="${ROOT:-$(cd "$here/../.." && pwd)}"
label="$1"; k="$2"; shift 2
julia_cmd=${JULIA:-julia}
project=${PROJECT:-$here}
out=${OUT:-$here/out/$label}
testfile=${TESTFILE:-$root/test/runtests.jl}
mkdir -p "$out"
cd "$root"
flags=("--heap-size-hint=${HEAP_HINT:-300M}" "$@")
for i in $(seq 1 "$k"); do
    if [ "${PKGTEST:-0}" = 1 ]; then
        jargs=$(printf '"%s",' "${flags[@]}")
        $julia_cmd --startup-file=no --project="$root" \
            -e "using Pkg; Pkg.test(julia_args=[${jargs%,}])" > "$out/run_$i.log" 2>&1 &
    else
        $julia_cmd --startup-file=no --check-bounds=yes "${flags[@]}" \
            --project="$project" "$testfile" > "$out/run_$i.log" 2>&1 &
    fi
done
wait
"$here/classify.sh" "$out"
