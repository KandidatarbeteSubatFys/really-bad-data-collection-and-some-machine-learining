This program extracts data from data files where the total deposited energy is more than a certain percentage of the total gun energy.
The program also removes some more data so that you get the same number of events per each number of guns used.

This program is used when you have your textfiles with data for each event of: the deposited energy for each crystal (XBe in the code),
the energy and cos(theta) for the guns used (gun in the code) and the sum of the deposited energy (XBsum in the code). XB stands for
crystal ball, but I can't think of anything that would limit this program to just the crystal ball detector. The format for every row 
in the data text files should be: 

XBe: Energy_1 Energy_2 ... Energy_n  (One energy for each crystal)
gun: Energy_1 cos(theta_1) Energy_2 cos(theta_2) ...  (If the maximum number of particles of all the events in the data
set is bigger than the number of guns used in an event, then the data for that event is filled up with zeros so that every row has the
same length.)
XBsum: Energy (Just the sum of XBe)

It is also IMPORTANT that the input text files are ordered from the lowest number of guns used to the highest number of guns used. So, for
example if your data consists of events with 1, 2 and 3 guns used, then the events where 1 gun is used should be first in the data files,
then should the event where 2 guns were used come and the events where 3 guns were used should come last.

The program has 7 compulsory arguments: percentage input_XBe.txt input_gun.txt input_XBsum.txt output_Xbe.txt output_gun.txt
output_XBsum.txt

The percentage argument is the percentage that the total deposited energy (XBsum) has to be larger than for the event to be saved.

Example: You have the three data files XBe.txt gun.txt XBsum.txt made for example from events between 1 and 5 guns used, and you want to
remove the events where the total deposited energy is less than 90 % of the gun energy and you want the same number of events per each
number of guns. Then you place the python script in the same as the files and place yourself in that folder and type:
python3 only_certain_percentage_dep_and_same_amount_for_each_number_of_guns.py 90 XBe.txt gun.txt XBsum.txt XBe_90%.txt gun_90%.txt 
XBsum_90%.txt

The names of the output files are optional, except don't use the same names as the input files.
