#!/usr/bin/bash

# This script is used to run the program
# ./build/LP_restart method ADMM restart 1 fixed_restart_length $i primal_weight_update 0 scaling 1 adaptive_step_size 0 tol -6 data_name qap10 &

method=ADMM
data_name=nug08-3rd

# define a array for fixed_restart_length
declare -a fixed_restart_length_array=(1024 4096 16384)
# declare -a fixed_restart_length_array=(16 64 256)
# declare -a fixed_restart_length_array=(65536 4096 16384)


max_parallel_jobs=5
count=0

for i in "${fixed_restart_length_array[@]}"
do
    ./build/LP_restart method $method restart 1 fixed_restart_length $i primal_weight_update 0 scaling 1 adaptive_step_size 1 tol -6 data_name $data_name &
    ((count++))
    if ((count == max_parallel_jobs)); then
        wait
        count=0
    fi
done

# Wait for any remaining jobs
wait