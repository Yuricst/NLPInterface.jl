#!/usr/bin/env bash
# Step 5: acceptance runs. The package-level reproducers (repro/patterns.jl)
# run K times each against every given checkout, interleaved in one batch, so
# machine load affects all checkouts alike.
#
#   dev/gc_corruption/acceptance.sh LABEL K CHECKOUT...
# Each CHECKOUT must have dev/gc_corruption/Project.toml instantiated.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
label="$1"; k="$2"; shift 2
variants=(${VARIANTS:-scope_worker worker_created_in_callback nested_testset})
out="$here/out/$label"; mkdir -p "$out"
n=${N:-150}
for i in $(seq 1 "$k"); do
    c=0
    for checkout in "$@"; do
        c=$((c + 1))
        for v in "${variants[@]}"; do
            ${JULIA:-julia} --startup-file=no --check-bounds=yes --heap-size-hint="${HEAP_HINT:-300M}" \
                --project="$checkout/dev/gc_corruption" "$here/repro/patterns.jl" "$v" "$n" \
                > "$out/c${c}_${v}_$i.log" 2>&1 &
        done
    done
    # Bound parallelism: one round (all checkouts x variants) at a time.
    wait
done
c=0
for checkout in "$@"; do
    c=$((c + 1))
    echo "checkout c$c: $checkout ($(git -C "$checkout" rev-parse --short HEAD 2>/dev/null)$(git -C "$checkout" diff --quiet 2>/dev/null || echo '+changes'))"
    for v in "${variants[@]}"; do
        bad=0; ok=0
        for f in "$out/c${c}_${v}"_[0-9]*.log; do
            if grep -qa "completed" "$f"; then ok=$((ok + 1)); else bad=$((bad + 1)); fi
        done
        printf "  %-28s aborts=%d ok=%d\n" "$v" "$bad" "$ok"
    done
done
