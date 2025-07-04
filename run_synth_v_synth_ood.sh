#!/bin/bash

# Defaults
OOD_SCENARIO=${1:-0}
PARALLEL_JOBS=${2:-3}

LOSS_FUNCS=("L1_Spec" "SIMSE_Spec" "DTW_Onset" "JTFS")
# LOSS_FUNCS=("JTFS")
COUNT=80

generate_commands() {
  for loss in "${LOSS_FUNCS[@]}"; do
    for ((i=1; i<=COUNT; i++)); do
      echo "python synth_v_synth_OOD.py --loss_fn $loss --learning_rate 0.045 --ood_scenario $OOD_SCENARIO"
    done
  done
}

generate_commands | parallel -j "$PARALLEL_JOBS"

