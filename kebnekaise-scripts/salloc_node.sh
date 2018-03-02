#!/bin/bash

#Arguments
#1: proj nr e.g. 2017-1-611
#2: CPU time in minutes
#3: nr of GPUs
#4: nr of tasks
#5: if you want whole node type --exclusive

#xterm -sb &
salloc -A SNIC$1 -t $2 --gres=gpu:k80:$3 -n $4 $5
