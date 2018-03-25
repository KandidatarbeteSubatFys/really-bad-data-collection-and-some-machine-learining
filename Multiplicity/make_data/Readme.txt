Makes training and evaluation batches for the multiplicity program. Makes the data from the crystal energy data and the
gun data in a npz file (for example make the npz file using superimpose_digitize_percentage.py), and saves the training
batches and the evaluation batches in a npz file. The input data has to be ordered from lowest number of guns used to
highest number of guns used. You can call the input file and the output file whatever you want, but it needs to end
with .npz

Example: You have made an npz file called superimposed_data_1_to_7_guns.npz using superimpose_digitize_percentage.py,
and you want to save the data in a npz file called multiplicity_data_set_superimposed_1_to_7_guns.npz. In the correct
folder, you type:
python3 multiplicity_data.py superimposed_data_1_to_7_guns.npz multiplicity_data_set_superimposed_1_to_7_guns.npz

Now you have made the training and evaluation batches and you're ready to use the gamma_multiplicity.py program.
