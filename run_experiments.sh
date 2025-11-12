#!/bin/bash

# Define the available loss functions
loss_functions=("L1_Spec" "DTW_Onset" "SIMSE_Spec" "JTFS")

# Define the learning rate (you can change this value as needed)
learning_rate=0.01

# Get the program ID from the first command-line argument
program_id=$1

# Get the number of runs from the second command-line argument
num_runs=$2

# Check if the program ID is provided and valid (0, 1, 2, or 3)
if [[ -z "$program_id" || "$program_id" -lt 0 || "$program_id" -gt 3 ]]; then
    echo "Usage: $0 <program_id> <num_runs> (valid program ID: 0, 1, 2, 3)"
    exit 1
fi

# Check if the number of runs is provided and is a positive integer
if [[ -z "$num_runs" || "$num_runs" -lt 1 ]]; then
    echo "Please provide a valid number of runs (greater than 0)."
    exit 1
fi

# Loop for the number of runs
for run in $(seq 1 $num_runs); do
    echo "Running experiment set $run of $num_runs"
    
    # Loop through all the loss functions
    for loss_fn in "${loss_functions[@]}"; do
        echo "Running experiment with loss function: $loss_fn, program ID: $program_id, learning rate: $learning_rate (Run $run)"
        
        # Run the Python program with the current loss function, learning rate, and program ID
        python random_experiment.py --loss_fn "$loss_fn" --learning_rate "$learning_rate" --program_id "$program_id"
        
        echo "Finished experiment with $loss_fn (Run $run)"
        echo "-----------------------------------------"
    done

    echo "Completed experiment set $run of $num_runs"
    echo "========================================="
done

echo "All experiments completed."

