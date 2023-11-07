#!/usr/bin/bash

# This script is used to run the program
# eg: ./build/LP_restart method ADMM restart 1 primal_weight_update 1 scaling 0 adaptive_step_size 0 tol -8 data_name n15-3

# define a string array for data name
# neos-506428 neos-932816 physiciansched6-2 n15-3 ns2118727
declare -a data_name_array=("neos-506428" "neos-932816" "physiciansched6-2" "n15-3" "ns2118727")

method=ADMM

max_parallel_jobs=5
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