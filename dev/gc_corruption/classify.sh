#!/usr/bin/env bash
# Classify the run_*.log files of a stress.sh output directory.
#   dev/gc_corruption/classify.sh OUTDIR
out="$1"; n=0; g=0; c=0; f=0; p=0
for log in "$out"/run_*.log; do
    n=$((n + 1))
    where=$(grep -a -m1 "in expression starting at" "$log" | sed 's|.*/||')
    if grep -qa "GC error (probable corruption)" "$log"; then
        g=$((g + 1))
        gcs=$(grep -a -m1 -o "GC: [0-9]*" "$log")
        vt=$(grep -a -m1 -A2 "GC error (probable corruption)" "$log" | sed -n 3p | cut -c1-60)
        echo "GCERR $(basename "$log")  $gcs  tag: $vt  during: $where"
    elif grep -qaE "signal [0-9]+" "$log"; then
        c=$((c + 1)); echo "CRASH $(basename "$log")  $(grep -a -m1 -oE "signal [0-9]+ \([0-9]+\): [A-Za-z ]+" "$log")  during: $where"
    elif grep -qaE "^NLPInterface +\||: completed" "$log" && ! grep -qaE "Test Failed|Error During Test|Some tests did not pass" "$log"; then
        p=$((p + 1)); echo "PASS  $(basename "$log")"
    else
        f=$((f + 1)); echo "FAIL  $(basename "$log")  $(grep -m1 -E 'Test Failed|Error During|ERROR:' "$log" | cut -c1-80)"
    fi
done
echo "SUMMARY $(basename "$out"): runs=$n gcerr=$g crash=$c fail=$f pass=$p"
