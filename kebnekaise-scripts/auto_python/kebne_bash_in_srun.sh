#!/bin/bash

#forgot to add !/bin/bash before, so if the program doesn't work, 
#try to remove it perhaps

#load modules
module load GCC/6.4.0-2.28  CUDA/9.0.176  OpenMPI/2.1.1
module load TensorFlow/1.5.0-Python-3.6.3 

#if we have one input argument, then run the python script. The if-
#statment isn't necessary, but doesn't hurt either (it's a remainder
#from an earlier version c: )
if [ $# -eq 1 ]
 then
python $1
fi
