#!/bin/bash
#Arguments
#1: time
#2: nr of GPUs
#3: For whole node, type: --exculsive (optional)
#4: python program to run

#allocates gpu nodes
salloc -A SNIC2017-1-611 --time=$1 --gres=gpu:k80:$2 --ntasks=1 $3 ./kebne_bash_salloc_to_srun.sh $4
