#!/bin/bash

squeue -o "%.12Q %.8i %.8j %.8u %.15a %.12l %.19S %.19V %.4D %.4C %.6h %R %m %b %f" --state=R -S "-p" | less
