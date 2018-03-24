This program makes a neural network that tries to predict the number of gamma fotons that interacts with the detector.
Should not be limited to one detector, should for example work on both crystal ball and dali2.

Steps on how to use it:
1) Simulate many single foton events in ggland (geant4) with sigma_e=0 (i.e try to not have any randomness/digitizer)
2) Use gamma_auto_home.py to convert the root file into three text files (crystal energies, gun data and total deposited
energy).
3) Use superimpose_digitize_percentage.py to superimpose the single foton events the way you want. It will add
sigma_e=5% after the superimposition is done. There is also an option to not use the foton events where less than for
example 90 % of the energy has been deposited in the detector.
4) Use multiplicity_data.py to create training and evaluation data from the superimposed data. The training data will
consist of zero events up to the highest number of particles used, and the evaluation data will only be based on events
where number of guns is between lowest number of guns used, to the second highest of guns used. So for example if your
superimposed data has events with 1 to 7 guns used, then you will train on 0 - 7 guns and evaluate on 1-6 guns.
5) Now you have the data for the program and you're ready to run the program. Open the program and specify the npz
input file, the number of hidden layers and the number of nodes per hidden layer.
