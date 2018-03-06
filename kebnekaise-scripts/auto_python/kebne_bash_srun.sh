#!/bin/bash
#Argument
#1: python script (optional)

#get jobID and get the digits before the decimal point (could be an
#issue if they change the number of digits in their job IDs)
jobID="$(sacct -b | awk END{print} | awk  '{print $1}')"
jobID=${jobID:0:7}

#runs the in_srun script and cancels the job when the script is finished
#running
srun --jobid=$jobID --pty bash -i kebne_bash_in_srun.sh $1
scancel $jobID
