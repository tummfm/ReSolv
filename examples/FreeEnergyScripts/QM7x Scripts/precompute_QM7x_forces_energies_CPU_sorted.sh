#!/bin/bash

#start=10000
#end=200000
#
#step=10000
#
##for ((i=start; i<=end || i<=end+step; i+=step)); do
#for((start=10000; start<=3800000; start+=200000)); do
#  for ((i=$start; i<=$start+190000; i+=step)); do
#    temp_i=$i
#    subtracted=$((i - 10000))
#
#    python Precompute_Amber_Energies_GPU.py $subtracted $temp_i &
#    echo "Iteration value (i): $temp_i, Subtracted value: $subtracted"
#  done
#  wait
#done

#start=3810000
#end=4000000
start=4010000
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
    python Precompute_Amber_Energies_GPU.py $subtracted $temp_i &
    echo "Iteration value (i): $temp_i, Subtracted value: $subtracted"
done


#start=10000
#end=3800000
#step=200000
#for ((start; start <= end; start += step)); do
#  for ((i = start; i <= start + 190000; i += step)); do
#    temp_i=$i
#    subtracted=$((i - 10000))
#
#    python Precompute_Amber_Energies_CPU.py $subtracted $temp_i &
#    echo "Iteration value (i): $temp_i, Subtracted value: $subtracted"
#  done
#  wait
#done
