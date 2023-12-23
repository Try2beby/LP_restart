#!/usr/bin/bash

# This script is used to run the program
# eg: ./build/LP_restart method ADMM restart 1 primal_weight_update 1 scaling 0 adaptive_step_size 0 tol -8 data_name n15-3

presolved_path=~/data_manage/cache/presolved/
# find all files in pgrk_path end with txt and save file name without extension into array
# declare -a data_name_array=($(find $pgrk_path -type f -name "graph_2*.txt" -printf "%f\n" | sed 's/.txt//g'))
# find all files in presolved_path end with .mps but not begin with pgrk and save file name without extension into array
declare -a data_name_array=($(find $presolved_path -type f -name "*.mps" -printf "%f\n" | sed 's/.mps//g' | grep -v "^pgrk"))

method=PDHG

max_parallel_jobs=10
count=0

# Array to hold the PIDs of the background jobs
declare -a pids=()

# This function will be called when the script is terminated
terminate_script() {
    echo "Terminating script..."
    # Kill each background job
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null
    done
    exit
}

# Catch the SIGINT and SIGTERM signals and call the terminate_script function
trap terminate_script SIGINT SIGTERM

# declare -a primal_weight_update_array=(1 0)
# declare -a scaling_array=(1 0)
declare -a primal_weight_update_array=(1)
declare -a scaling_array=(1)

printf "%s\n" "${primal_weight_update_array[@]}" | while read pau; do
    printf "%s\n" "${scaling_array[@]}" | while read sc; do
        printf "%s\n" "${data_name_array[@]}" | xargs -I {} -P $max_parallel_jobs sh -c './build/LP_restart method $0 restart 1 primal_weight_update $1 scaling $2 adaptive_step_size 0 tol -8 data_name $3 & pid=$!; echo $pid; wait $pid' $method $pau $sc {} | while read pid; do
            pids+=("$pid")
        done
    done
done

# Wait for any remaining jobs
wait