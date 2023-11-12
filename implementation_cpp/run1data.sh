#!/usr/bin/bash

# This script is used to run the program
# ./build/LP_restart method ADMM restart 1 fixed_restart_length $i primal_weight_update 0 scaling 1 adaptive_step_size 0 tol -6 data_name qap10 &

method=ADMM
declare -a data_name_array=(qap10 qap15 nug20 nug08-3rd)

# define a array for fixed_restart_length
declare -a fixed_restart_length_array_1=(1024 4096 16384)
declare -a fixed_restart_length_array_3=(16 64 256)
declare -a fixed_restart_length_array_4=(65536 4096 16384)

declare -A matrix
num_rows=4
num_cols=3
matrix[1]=${fixed_restart_length_array_1[@]}
matrix[2]=${fixed_restart_length_array_1[@]}
matrix[3]=${fixed_restart_length_array_3[@]}
matrix[4]=${fixed_restart_length_array_4[@]}

declare -a scaling_array=(1 0)

max_parallel_jobs=5
count=0

row_index=0
for i in "${data_name_array[@]}"
do 
    for sc in "${scaling_array[@]}"
    do
        for ((col_index=0; col_index<num_cols; col_index++))
        do
            ./build/LP_restart method $method restart 1 fixed_restart_length ${matrix[$row_index,$col_index]} primal_weight_update 0 scaling $sc adaptive_step_size 0 tol -6 data_name $i eta 100 &
            ((count++))
            if ((count == max_parallel_jobs)); then
                wait
                count=0
            fi
        done
    done
    ((row_index++))
done

# Wait for any remaining jobs
wait