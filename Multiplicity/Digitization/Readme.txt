This program takes in many single guns events from ggland (for crystal ball: sigma_e=0, or equvivalenty for other detectors with as little
digitization as possible) and returns superimposed events and apply sigma_e=5% for each crystal energy after the superimposition.
The benefit with superimposing data from single gun events is that all fired guns will interact with the detector. 

Example: You want to create events ranging from 1 gun to 7. You only want to build these events on single gun event that deponated at
least 90 % of their energy to the detector. Then type:
python3 superimposed_digitalisation_certain_percentage.py 1 7 90 inputfile_crystal_energies.txt inputfile_gun.txt output_crystal_energies.txt output_gun.txt output_XBesum.txt

You will always get the same amount of events for each number of particles.

Tip 1: Set percentage=0 for no requirement on deposited energy
Tip 2: Let say you want x event for each number of particles. Then you will have to simulate
(desired_number_of_events)*sum_(k=lowest_number_of_particles)^(highest_number_of_particles) k/lowest_number_of_particles
amount of single gun events.
