#!/bin/bash

LOSS_FUNCS=("L1_Spec" "SIMSE_Spec" "DTW_Onset" "JTFS")
COUNT=40

generate_commands() {
  for loss in "${LOSS_FUNCS[@]}"; do
    for ((i=1; i<=COUNT; i++)); do
      echo "python synth_v_synth_OOD.py --loss_fn $loss --learning_rate 0.045 --ood_scenario 0"
    done
  done
}

generate_commands | parallel -j 3
