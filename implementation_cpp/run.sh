#!/usr/bin/bash

# This script is used to run the program
# ./build/LP_restart -m PDHG -d 3 -r [0,1] -l [-1,[16384,65536,262144]]

dataidx=3
# method=PDHG
method=ADMM

# ./build/LP_restart -m ${method} -d ${dataidx} -r 0 -l -1
# ./build/LP_restart -m ${method} -d ${dataidx} -r 1 -l -1

declare -a restart_length_array
# PDHG
# restart_length_array=(4096 16384 65536)
# restart_length_array=(64 256 1024)
# restart_length_array=(16384 65536 262144)

# ADMM
# restart_length_array=(1024 4096 16384)
# restart_length_array=(16 64 256)
restart_length_array=(4096 16384 65536)

max_parallel_jobs=4
count=0

for i in "${restart_length_array[@]}"
do
    ./build/LP_restart -m ${method} -d ${dataidx} -r 1 -l $i &
    ((count++))
    if ((count == max_parallel_jobs)); then
        wait
        count=0
    fi
done

# Wait for any remaining jobs
wait