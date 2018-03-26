import sys
import os
import numpy as np


def main():

    # number_of_partices=number of maximum guns used in any file
    number_of_particles = sys.argv[1]
    detector=sys.argv[2]
    root_files = ["" for x in range(len(sys.argv) - 6)]
    output_files = ["" for x in range(3)]
    for i in range((len(sys.argv) - 6)):
        root_files[i] = sys.argv[i + 3]
    # output files are made. The first is for the XBe data, the second for the gun
    # data and the third is for the XBEsum data
    for i in range(3):
        output_files[i] = sys.argv[(len(sys.argv) - 3 + i)]
        if os.path.isfile(output_files[i]):
            # clears previous output files
            open(output_files[i], 'w').close()

    for i in range((len(sys.argv) - 6)):
        # converts one root file to txt
        make_txt(root_files[i],detector)
        # adds right amount of zeros to gunfile. Comment this line and do one
        # modification to the make_txt function as well to not add zeros.
        add_zeros_and_shuffle_gun(number_of_particles)
        # appends the data from one root file to final output files
        add_to_final_files(output_files)

    # convers one root file to three txt files


def make_txt(file,detector):
    # h102 class is made from terminal
    os.system("root -q \'make_class.C(\"" + file + "\")\'")

    if detector=='XB':
        file_name="XB_h102_backup.C"
    elif detector=='dali2':
        file_name="dali2_h102_backup.C"
    else:
        raise ValueError("The second argument must either be XB or dali2")
    # the code in the default h102.C class is replaced
    with open(file_name) as f:
        with open("h102.C", "w") as f1:
            for line in f:
                f1.write(line)
    # the Loop() method is run in root from terminal
    os.system("root -q root_loop.C")


# add zeros to gun data depending on the number of particles
def add_zeros_and_shuffle_gun(number_of_particles):
    with open("gun_data.txt") as f:
        with open("tempfile.txt", "w") as f1:
            for line in f:
                # line includes \n so removes it with rstrip()
                line = line.rstrip()
                string_list = line.split(' ')
                f1.write(line + (3 * int(number_of_particles) - len(string_list)) * " 0" + " \n")
                # f1.write(line + (2 * int(number_of_particles) - len(tmp_string_list)) * " 0" + " \n")


# adds the final data from one root file to the final ouput files
def add_to_final_files(output_files):
    # appends the XBe data
    with open("crystal_energies.txt") as f:
        with open(output_files[0], "a") as f1:
            for line in f:
                f1.write(line)
    # appends the gun data
    with open("tempfile.txt") as f:  # change to gamma_gunTVals if not adding zeros
        with open(output_files[1], "a") as f1:
            for line in f:
                f1.write(line)
    # appends the XBEsum data
    with open("sum_of_dep_energies.txt") as f:
        with open(output_files[2], "a") as f1:
            for line in f:
                f1.write(line)


if __name__ == "__main__":
    main()

