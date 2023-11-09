#!/usr/bin/bash

# This script is used to run the program
# eg: ./build/LP_restart method ADMM restart 1 primal_weight_update 1 scaling 0 adaptive_step_size 0 tol -8 data_name n15-3

presolved_path=./data/cache/presolved/
# find all files in presolved_path and save file name without extension into array
declare -a data_name_array=($(find $presolved_path -type f -printf "%f\n" | cut -d. -f1))
method=PDHG

max_parallel_jobs=10
count=0

for i in "${data_name_array[@]}"
do
    ./build/LP_restart method $method restart 1 primal_weight_update 1 scaling 0 adaptive_step_size 0 tol -8 data_name $i &
    ((count++))
    if ((count == max_parallel_jobs)); then
        wait
        count=0
    fi
done

# Wait for any remaining jobs
wait