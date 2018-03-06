#!/bin/bash

#Arguments
#1: time
#2: nr of GPUs
#3: For whole node, type: --exculsive (optional)
#4: python script

#opens salloc script in xterm terminal
xterm -sb -e "./kebne_bash_salloc.sh \"$1\" \"$2\" \"$3\" \"$4\";bash" &
