#!/bin/bash

# $1 is root file from simulation with DALI

# This code is for DALI only

root -q 'root_make_class_and_gen_data_files_DALI.C('\"$1\"')'
echo "ROOT finished"
echo "Removing h102.C and h102.h"
rm h102.C
rm h102.h
