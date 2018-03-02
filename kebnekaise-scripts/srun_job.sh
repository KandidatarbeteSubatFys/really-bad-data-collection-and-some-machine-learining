#!/bin/bash

#jobID is aquired after allocating a node

sacct -b | awk END{print} | awk '{print $1}'
jobID=$(!!) 

srun --jobid=$jobID --pty bash -i
