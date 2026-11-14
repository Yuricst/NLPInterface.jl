#!/usr/bin/env bash
# Run test/runtests.jl once under gdb; on "GC error (probable corruption)" stop
# in gc_dump_queue_and_abort and dump every thread's backtrace and the bad tag.
#
#   dev/gc_corruption/gdb_run.sh OUTFILE [extra julia flags...]
# Same environment knobs as stress.sh (JULIA must be a plain binary path here).
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="${ROOT:-$(cd "$here/../.." && pwd)}"
outfile="$(realpath -m "$1")"; shift
julia_bin=${JULIA:-$(readlink -f "$(command -v julia)")}
cd "$root"
gdb -q -batch \
    -ex "set pagination off" -ex "set print thread-events off" \
    -ex "set breakpoint pending on" \
    -ex "handle SIGSEGV nostop noprint pass" -ex "handle SIGUSR2 nostop noprint pass" \
    -ex "handle SIGINT nostop noprint pass" -ex "handle SIGPIPE nostop noprint pass" \
    -ex "break gc_dump_queue_and_abort" \
    -ex "run" \
    -ex "echo \n=== bad tag (vt) and caller frames ===\n" \
    -ex "info registers rdi rsi" \
    -ex "frame 1" -ex "info locals" -ex "frame 2" -ex "info locals" \
    -ex "echo \n=== memory around the bad object (vt = rsi) ===\n" -ex 'x/16gx $rsi-64' \
    -ex "echo \n=== mappings ===\n" -ex "info proc mappings" \
    -ex "echo \n=== all threads ===\n" -ex "info threads" -ex "thread apply all bt 60" \
    -ex "kill" \
    --args "$julia_bin" --startup-file=no --check-bounds=yes \
        --heap-size-hint="${HEAP_HINT:-300M}" --project="${PROJECT:-$here}" "$@" \
        "${TESTFILE:-$root/test/runtests.jl}" > "$outfile" 2>&1
grep -c "gc_dump_queue_and_abort" "$outfile" > /dev/null && echo "hit: $(grep -c 'Breakpoint 1,' "$outfile")" || true
