#!/bin/bash

#runs srun script in new xterm terminal
xterm -sb -e "./kebne_bash_srun.sh $1;bash" 
