#!/bin/bash

#start=10000
#end=2000000
start=2010000
end=4195237
step=10000

for ((i=start; i<=end || i<=end+step; i+=step)); do
    if ((i == 4200000)); then
        temp_i=4195237
        subtracted=4190000
    else
        temp_i=$i
        subtracted=$((i - 10000))
    fi
    python Precompute_Amber_Energies_GPU.py $subtracted $temp_i &&
    echo "Iteration value (i): $temp_i, Subtracted value: $subtracted"
done