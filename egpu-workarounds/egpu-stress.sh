#!/usr/bin/env bash
#
# egpu-stress.sh — live eGPU stability dashboard.
#
# Repeatedly runs the CUDA test suite in the background while showing a live,
# in-place terminal readout of power / SM clock / temperature (current + running
# min/avg/max), a spinner status indicator, run/pass/regression counters, and
# prominent detection if the GPU falls off the bus.
#
# Usage:
#   ./egpu-stress.sh                 # loop the suite + live dashboard
#   ./egpu-stress.sh --monitor       # dashboard only, do NOT run the suite
#   ./egpu-stress.sh --interval 2    # telemetry refresh seconds (default 1)
#
# Ctrl-C to quit (restores the cursor and stops the background suite loop).
#
set -uo pipefail

INTERVAL=1
RUN_SUITE=1
for a in "$@"; do
    case "$a" in
        --monitor)  RUN_SUITE=0 ;;
        --interval) shift; : ;;                       # handled below
    esac
done
# simple --interval N parse
prev=""
for a in "$@"; do
    [[ "$prev" == "--interval" ]] && INTERVAL="$a"
    prev="$a"
done

# ---- locate the test suite -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_SCRIPT="${TESTS_SCRIPT:-}"
if [[ -z "$TESTS_SCRIPT" ]]; then
    for cand in "$SCRIPT_DIR/../tests/run_tests.sh" \
                "/home/plotfi/opt/dev/tensara-kernels/tests/run_tests.sh"; do
        [[ -x "$cand" ]] && { TESTS_SCRIPT="$cand"; break; }
    done
fi
if [[ $RUN_SUITE -eq 1 && ! -x "${TESTS_SCRIPT:-/nonexistent}" ]]; then
    echo "run_tests.sh not found — falling back to --monitor. Set TESTS_SCRIPT=… to override." >&2
    RUN_SUITE=0
fi

# ---- shared state ----------------------------------------------------------
STATE_DIR="$(mktemp -d /tmp/egpu-stress.XXXXXX)"
SUITE_STATE="$STATE_DIR/suite"          # "runs|passed|regressions|builderr|status"
echo "0|-|-|-|idle" > "$SUITE_STATE"

cleanup() {
    [[ -n "${SUITE_PID:-}" ]] && kill "$SUITE_PID" 2>/dev/null
    tput cnorm 2>/dev/null            # restore cursor
    tput sgr0 2>/dev/null
    rm -rf "$STATE_DIR"
    printf '\n'
    exit 0
}
trap cleanup INT TERM EXIT

# ---- background: loop the suite forever ------------------------------------
suite_loop() {
    local n=0
    while true; do
        n=$((n+1))
        # mark running (preserve last counts)
        IFS='|' read -r _ lp lr lb _ < "$SUITE_STATE"
        echo "$n|$lp|$lr|$lb|running" > "$SUITE_STATE"
        out="$("$TESTS_SCRIPT" 2>&1)"
        p=$(sed -n 's/.*passed: *\([0-9]*\).*/\1/p'      <<<"$out" | head -1)
        r=$(sed -n 's/.*regressions: *\([0-9]*\).*/\1/p' <<<"$out" | head -1)
        b=$(sed -n 's/.*build errors: *\([0-9]*\).*/\1/p'<<<"$out" | head -1)
        echo "$n|${p:-?}|${r:-?}|${b:-?}|done" > "$SUITE_STATE"
    done
}
if [[ $RUN_SUITE -eq 1 ]]; then
    suite_loop &
    SUITE_PID=$!
fi

# ---- dashboard -------------------------------------------------------------
GREEN=$'\033[1;32m'; RED=$'\033[1;31m'; YEL=$'\033[1;33m'; CYA=$'\033[1;36m'
DIM=$'\033[2m'; BOLD=$'\033[1m'; RST=$'\033[0m'
SPIN='|/-\'; si=0
start=$(date +%s)
samples=0; drops=0
pmin=; pmax=; psum=0
cmin=; cmax=; csum=0
tmin=; tmax=; tsum=0

upd() {  # upd <var-min> <var-max> <sumvar> <value>
    local __min=$1 __max=$2 __sum=$3 v=$4
    local cur_min=${!__min} cur_max=${!__max} cur_sum=${!__sum}
    [[ -z "$cur_min" || $(echo "$v < $cur_min" | bc -l) -eq 1 ]] && cur_min=$v
    [[ -z "$cur_max" || $(echo "$v > $cur_max" | bc -l) -eq 1 ]] && cur_max=$v
    cur_sum=$(echo "$cur_sum + $v" | bc -l)
    printf -v "$__min" '%s' "$cur_min"
    printf -v "$__max" '%s' "$cur_max"
    printf -v "$__sum" '%s' "$cur_sum"
}

bar() {  # bar <value> <max> <width>  -> filled bar string
    local v=$1 mx=$2 w=$3
    local n=$(printf '%.0f' "$(echo "$v/$mx*$w" | bc -l)")
    (( n<0 )) && n=0; (( n>w )) && n=w
    printf '%s' "$(printf '%*s' "$n" '' | tr ' ' '#')$(printf '%*s' "$((w-n))" '')"
}

tput civis 2>/dev/null      # hide cursor
clear

while true; do
    si=$(( (si+1) % 4 ))
    now=$(date +%s); elapsed=$((now-start))
    hh=$((elapsed/3600)); mm=$(((elapsed%3600)/60)); ss=$((elapsed%60))

    if reading=$(nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu \
                    --format=csv,noheader,nounits 2>/dev/null); then
        IFS=',' read -r pw sm tp ut <<<"$reading"
        pw=$(echo "$pw"|tr -d ' '); sm=$(echo "$sm"|tr -d ' ')
        tp=$(echo "$tp"|tr -d ' '); ut=$(echo "$ut"|tr -d ' ')
        samples=$((samples+1))
        upd pmin pmax psum "$pw"
        upd cmin cmax csum "$sm"
        upd tmin tmax tsum "$tp"
        pavg=$(echo "scale=1;$psum/$samples"|bc); cavg=$(echo "$csum/$samples"|bc)
        tavg=$(echo "$tsum/$samples"|bc)
        status="${GREEN}● UP${RST}"
        gpu_ok=1
    else
        drops=$((drops+1)); gpu_ok=0
        status="${RED}✖ DROPPED OFF BUS${RST}"
    fi

    IFS='|' read -r runs sp sr sb sst < "$SUITE_STATE"
    case "$sst" in
        running) rstat="${YEL}running${RST}" ;;
        done)    rstat="${GREEN}done${RST}"  ;;
        *)       rstat="${DIM}idle${RST}"    ;;
    esac
    [[ "${sr:-0}" =~ ^[0-9]+$ && "${sr:-0}" -gt 0 ]] && rcol="$RED" || rcol="$GREEN"
    [[ "${sb:-0}" =~ ^[0-9]+$ && "${sb:-0}" -gt 0 ]] && bcol="$RED" || bcol="$GREEN"

    tput cup 0 0
    printf "%s┌─ eGPU STRESS + TELEMETRY ─ %c ─────────────────────────────┐%s\033[K\n" "$CYA" "${SPIN:$si:1}" "$RST"
    printf "  status: %b   uptime: %02d:%02d:%02d   samples: %d   drops: %s%d%s\033[K\n" \
        "$status" "$hh" "$mm" "$ss" "$samples" "$([[ $drops -gt 0 ]] && echo "$RED" || echo "$GREEN")" "$drops" "$RST"
    printf '\033[K\n'
    if [[ $gpu_ok -eq 1 ]]; then
        printf "  %bPOWER%b  %6.1f W   [%s]  min %s  avg %s  max %s\033[K\n" \
            "$BOLD" "$RST" "$pw" "$(bar "$pw" 150 24)" "$pmin" "$pavg" "$pmax"
        printf "  %bCLOCK%b  %6s MHz [%s]  min %s  avg %s  max %s\033[K\n" \
            "$BOLD" "$RST" "$sm" "$(bar "$sm" 1500 24)" "$cmin" "$cavg" "$cmax"
        printf "  %bTEMP %b  %6s °C  [%s]  min %s  avg %s  max %s\033[K\n" \
            "$BOLD" "$RST" "$tp" "$(bar "$tp" 100 24)" "$tmin" "$tavg" "$tmax"
        printf "  %bUTIL %b  %6s %%\033[K\n" "$BOLD" "$RST" "$ut"
    else
        printf "  %b*** nvidia-smi reports no device — GPU is off the bus ***%b\033[K\n" "$RED" "$RST"
        printf '\033[K\n\033[K\n\033[K\n'
    fi
    printf '\033[K\n'
    if [[ $RUN_SUITE -eq 1 ]]; then
        printf "  %bSUITE%b  run #%s (%b)  passed %b%s%b  reg %b%s%b  build-err %b%s%b\033[K\n" \
            "$BOLD" "$RST" "${runs:-0}" "$rstat" \
            "$GREEN" "${sp:--}" "$RST" "$rcol" "${sr:--}" "$RST" "$bcol" "${sb:--}" "$RST"
    else
        printf "  %bSUITE%b  (monitor-only mode)\033[K\n" "$DIM" "$RST"
    fi
    printf "%s└──────────────────────────────────────────────────────────────┘%s\033[K\n" "$CYA" "$RST"
    printf "  %bCtrl-C to quit%s\033[K\n" "$DIM" "$RST"
    printf '\033[J'      # clear to end of screen

    sleep "$INTERVAL"
done
