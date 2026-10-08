#!/usr/bin/env bash
# Live view of the Sun et al. (2024) ablation sweep. Read-only.
set -u
cd /home/sandy/Projects/PEP-26 || exit 1
PROJ=runs/ablation
TARGET=150

echo -e "\033[1m  Sun et al. (2024) — YOLOv8 ablation sweep\033[0m"
echo -e "  RDD2022 Chinese subset  |  150 epochs, batch 4, SGD lr0 0.01  |  paper Table 2 targets"
echo

while true; do
    clear
    printf "  \033[1m%-11s %8s %9s %10s  %-25s %s\033[0m\n" arm epochs mAP50 mAP50-95 progress "paper mAP50"
    printf "  %-11s %8s %9s %10s  %-25s  %s\n" "----------" "-------" "--------" "---------" "----------------------" "------------"

    running=$(pgrep -af "_worker.py" 2>/dev/null)
    for arm in baseline A_spd B_neck C_faster D_wiou E_all; do
        f="$PROJ/$arm/results.csv"
        # paper Table 2 targets, for reference next to the live numbers
        case $arm in
            baseline) tgt="0.746";; A_spd) tgt="0.766";; B_neck) tgt="0.764";;
            C_faster) tgt="0.744";; D_wiou) tgt="0.758";; E_all) tgt="0.774";;
        esac
        if [ -f "$f" ]; then
            n=$(( $(wc -l < "$f") - 1 ))
            m=$(tail -1 "$f" | cut -d, -f8)
            m95=$(tail -1 "$f" | cut -d, -f9)
            m=$(printf "%.4f" "${m:-0}" 2>/dev/null || echo "-")
            m95=$(printf "%.4f" "${m95:-0}" 2>/dev/null || echo "-")
            if echo "$running" | grep -q "_worker.py $arm "; then st="● live"; else st="○ idle"; fi
            bar=$(printf "%*s" $(( n * 20 / TARGET )) '' | tr ' ' '#')
            bar="$bar$(printf "%*s" $(( 20 - ${#bar} )) '' | tr ' ' '.')"
            printf "  %-11s %8s %9s %10s  %-25s %s\n" "$arm" "$n/150" "$m" "$m95" "$bar $st" "$tgt"
        else
            printf "  %-11s %8s %9s %10s  %-25s %s\n" "$arm" "—" "—" "—" "........................." "$tgt"
        fi
    done

    echo
    if pgrep -f "run_ablation.py" >/dev/null 2>&1; then
        printf "  runner: \033[32malive\033[0m   workers: %s\n" "$(pgrep -fc '_worker.py' 2>/dev/null || echo 0)"
    else
        printf "  runner: \033[31mGONE\033[0m\n"
    fi
    printf "  gpu: %s   ram-avail: %s\n" \
        "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)" \
        "$(free -g | awk '/^Mem:/{print $7" GiB"}')"

    if [ -f "$PROJ/sweep2.log" ]; then
        echo
        echo "  ---- last sweep events ----"
        tail -3 "$PROJ/sweep2.log" | sed 's/^/  /'
    fi
    echo
    echo "  Ctrl-C closes this window (training is unaffected)."
    sleep 10
done
