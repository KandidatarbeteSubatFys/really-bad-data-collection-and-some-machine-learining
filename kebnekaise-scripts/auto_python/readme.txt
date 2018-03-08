This program runs a python script on kebnekaise (only for single core). The job is canceled after the python-script
is finished.

Preferably stand in the pfs folder.

IMPORTANT: Make sure that you have permission to execute all files.

Example: Want to run python script test.py that is expected to take less than 3 hours and you don't need more than one
         k80 graphics-card to run on.
         Solution: in pfs folder, type: ./kebne_bash_start.sh 180 1 "" test.py
         
Arguments: 1: time in minutes
           2: the number of k80 cards allocated (only reasonable choice thus far is 1, since the program only handles
              single core programs.
           3: if you want an exclusive node, write --exclusive. Otherwise write ""
           4: name of python script
