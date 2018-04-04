#!/bin/bash

# $1 is root file from simulation with XB

# This code is for XB only

root -q 'root_make_class_and_gen_data_files_XB.C('\"$1\"')'
echo "ROOT finished"
echo "Removing h102.C and h102.h"
rm h102.C
rm h102.h
