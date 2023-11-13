#!/usr/bin/bash

# This script is used to run the program
# eg: ./build/LP_restart method ADMM restart 1 primal_weight_update 1 scaling 0 adaptive_step_size 0 tol -8 data_name n15-3

presolved_path=./data/cache/presolved/
pgrk_path=./data/pagerank/
# find all files in presolved_path end with .mps and save file name without extension into array
# declare -a data_name_array=($(find $presolved_path -type f -name "*.mps" -printf "%f\n" | sed 's/.mps//g'))
# find all files in pgrk_path end with txt and save file name without extension into array
declare -a data_name_array=($(find $pgrk_path -type f -name "graph_2*.txt" -printf "%f\n" | sed 's/.txt//g'))

method=PDHG

max_parallel_jobs=10
count=0

# This function will be called when the script is terminated
terminate_script() {
    echo "Terminating script..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Catch the SIGINT and SIGTERM signals and call the terminate_script function
trap terminate_script SIGINT SIGTERM

declare -a primal_weight_update_array=(1 0)
declare -a scaling_array=(1 0)

for pau in "${primal_weight_update_array[@]}"
do
    for sc in "${scaling_array[@]}"
    do
        for i in "${data_name_array[@]}"
        do
            ./build/LP_restart method $method restart 1 primal_weight_update $pau scaling $sc adaptive_step_size 0 tol -8 data_name $i &
            ((count++))
            if ((count == max_parallel_jobs)); then
                wait
                count=0
            fi
        done
    done
done

# Wait for any remaining jobs
wait