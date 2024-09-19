#!/bin/bash

# Check if program number is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <program_number>"
    exit 1
fi

PROGRAM_NUMBER=$1
LOSS_FUNCTIONS=("L1_Spec" "SIMSE_Spec" "DTW_Onset" "JTFS")

for LOSS_FN in "${LOSS_FUNCTIONS[@]}"; do
    echo "Running with loss function: $LOSS_FN and program number: $PROGRAM_NUMBER"
    python3 inside_search_experiment.py --loss_fn "$LOSS_FN" --program_number "$PROGRAM_NUMBER"
done
