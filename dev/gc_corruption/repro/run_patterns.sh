#!/usr/bin/env bash
# Run every variant of patterns.jl K times in parallel and classify.
#   dev/gc_corruption/repro/run_patterns.sh LABEL K N [PROJECT] [julia flags...]
# SCRIPT selects the script (default patterns.jl), VARIANTS the variants.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
label="$1"; k="$2"; n="$3"; project="${4:-$here/..}"; shift 4 2>/dev/null || shift $#
out="$here/../out/$label"; mkdir -p "$out"
variants=(${VARIANTS:-plain spawn_outside spawn_in_callback_trivial spawn_in_callback_nested worker_solve spawn_in_callback_ipopt})
for v in "${variants[@]}"; do
    for i in $(seq 1 "$k"); do
        ${JULIA:-julia} --startup-file=no --check-bounds=yes --heap-size-hint="${HEAP_HINT:-300M}" \
            --project="$project" "$@" "$here/${SCRIPT:-patterns.jl}" "$v" "$n" > "$out/${v}_$i.log" 2>&1 &
    done
done
wait
for v in "${variants[@]}"; do
    g=0; c=0; p=0; o=0
    for f in "$out/${v}"_[0-9]*.log; do
        if grep -qa "GC error (probable corruption)" "$f"; then g=$((g+1))
        elif grep -qaE "signal [0-9]+" "$f"; then c=$((c+1))
        elif grep -qa "completed" "$f"; then p=$((p+1))
        else o=$((o+1)); fi
    done
    printf "%-28s gcerr=%d crash=%d ok=%d other=%d\n" "$v" "$g" "$c" "$p" "$o"
done
