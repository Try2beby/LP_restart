#!/usr/bin/bash

# This script is used to run the program
# ./build/LP_restart -m PDHG -d 3 -r [0,1] -l [-1,[16384,65536,262144]]

dataidx=3
method=PDHG

./build/LP_restart -m ${method} -d ${dataidx} -r 0 -l -1
./build/LP_restart -m ${method} -d ${dataidx} -r 1 -l -1

declare -a restart_length_array
# restart_length_array=(4096 16384 65536)
# restart_length_array=(64 256 1024)
restart_length_array=(16384 65536 262144)

for i in "${restart_length_array[@]}"
do
    ./build/LP_restart -m ${method} -d ${dataidx} -r 1 -l $i
done
