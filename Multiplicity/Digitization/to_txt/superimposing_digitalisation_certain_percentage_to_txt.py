import sys
import numpy as np


# Reads data from a space delimited text file of floats and stores the data in a numpy array
# Do not now if there is a faster way, but this works pretty ok at least.
def read_data_to_numpy(file,rows,cols):
    out=np.zeros((rows,cols))
    with open(file, 'r') as f:
        index=0
        for lines in f:
            if index%100000==0:
                print("Number of read rows: " + str(index))
            line = lines.rstrip()
            out[index]=np.fromstring(line,dtype=np.float32,sep=' ')
            index=index+1
    return out


# function used for getting multiple rows from a matrix (numpy format matrix, not list).
def get_rows_in_interval_numpy(numpy_matrix,start_row_index, end_row_index):
    if end_row_index==-1:
        end_row_index=len(numpy_matrix)-1
    out=np.zeros((end_row_index-start_row_index+1,len(numpy_matrix[0])))
    index=0
    for i in range(start_row_index,end_row_index+1):
        out[index]=numpy_matrix[i]
        index=index+1
    return out


# adds multiple rows of same size to one
def add_rows_to_one(matrix):
    sum_row=matrix.sum(axis=0)
    return sum_row


# Takes many singe gun-events and puts them next to each other, as well as adding the correct number of zeros
# so that every row is the same length.
def flatten_and_add_zeros(matrix,highest_number_of_particles):
    flat_list=matrix.flatten()
    flat_list=np.concatenate((flat_list,np.zeros(2*highest_number_of_particles-len(flat_list))),axis=0)
    return flat_list


# this function gives the superimposed events some randomness. For each crystal energy, the energy is used as expected value of
# a normaldistribution and the variance is 5% of the energy.
def crystal_energies_sigma_5_percent(superimposed_crystal_energy_list):
    for i in range(len(superimposed_crystal_energy_list)):
        if superimposed_crystal_energy_list[i]!=0:
            superimposed_crystal_energy_list[i]=np.random.normal(superimposed_crystal_energy_list[i],0.05*superimposed_crystal_energy_list[i],1)
    return superimposed_crystal_energy_list


# Returns the amount of rows of a file in an efficient way
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# Removes singe gun events that do not fulfill that the total deposited energy should be greater or equal to the
# specified percentage times the gun energy
def only_certain_percentage(crystal_input_matrix,gun_input_matrix,percentage):
    # Make numpy matrices of the same shape as the input matrices
    crystal_out=np.zeros(crystal_input_matrix.shape,dtype=np.float32)
    gun_out = np.zeros(gun_input_matrix.shape,dtype=np.float32)

    index=0
    for i in range(len(crystal_input_matrix)):
        if i%100000==0:
            print("Number of rows checked for tot_dep<90%gun_dep: "+str(i))
        # if the percentage requirement is fulfilled, then this event is saved
        if sum(crystal_input_matrix[i])>=percentage/100*gun_input_matrix[i][0]:
            crystal_out[index]=crystal_input_matrix[i]
            gun_out[index]=gun_input_matrix[i]
            index=index+1
    # Here you get the index where the unused rows of zeros starts
    for i in range(len(gun_out)):
        if gun_out[len(gun_out)-1-i,0]!=0:
            index_of_first_zero_row=len(gun_out)-i
            break
    # And then you return the matriced with the zero-rows deleted
    return np.delete(crystal_out,[i for i in range(index_of_first_zero_row,len(crystal_out))],axis=0), np.delete(gun_out,[i for i in range(index_of_first_zero_row,len(gun_out))],axis=0)



def main(lowest_number_of_particles, highest_number_of_particles, percentage,crystal_energy_input_file, gun_input_file, crystal_energy_output_file, gun_output_file, total_dep_energy_output_file):
    print("Time example: takes around 10 minutes for 10 000 000 events for particles from 1 to 7 on pclab-232")

    print("Reading data")
    number_of_crystals = 162
    print("Reading number of rows in files")
    number_of_events = file_len(gun_input_file)
    print("Reading the crystal energy file")
    crystal_energy_input_matrix = read_data_to_numpy(crystal_energy_input_file, number_of_events, number_of_crystals)
    print("Reading the gun file")
    gun_input_matrix = read_data_to_numpy(gun_input_file,number_of_events,2)

    print("Removing events where (total depositeed energy)<percentage*(gun_energy)")
    # Removes events where less than the desired percentage was deposited
    if percentage!=0:
        crystal_energy_input_matrix, gun_input_matrix=only_certain_percentage(crystal_energy_input_matrix,gun_input_matrix,percentage)

    number_of_events=len(crystal_energy_input_matrix)

    # Here the events needed for the lowest number of guns is calculated based on the lowest and highest number of
    # particles and on the amount of single gun events
    events_needed_for_lowest_number_of_particles=number_of_events/sum([i/lowest_number_of_particles for i in range(lowest_number_of_particles,highest_number_of_particles+1)])
    events_needed_for_lowest_number_of_particles=int(events_needed_for_lowest_number_of_particles-events_needed_for_lowest_number_of_particles%lowest_number_of_particles)

    print("Start superimposing and digitising as well as writing to file")
    with open(crystal_energy_output_file,"w") as f_cr:
        with open(gun_output_file, "w") as f_gun:
            with open(total_dep_energy_output_file, "w") as f_sum:

                # for each number of particles, the events are created and written to the output files
                for i in range (lowest_number_of_particles,highest_number_of_particles+1):
                    # for each number of particles, there is a start and end row-index of the input data matrixes that specify
                    # which data should be used for creating the events with the current number of particles.
                    start_row_index=int(events_needed_for_lowest_number_of_particles/lowest_number_of_particles*sum([j for j in range(lowest_number_of_particles,i)]))
                    end_row_index = int(events_needed_for_lowest_number_of_particles/lowest_number_of_particles*sum([j for j in range(lowest_number_of_particles,i+1)]))-1
                    step=i-1
                    # Then all the rows that will make up the events for the current number of particles are being
                    # swept over
                    for j in range(start_row_index,end_row_index+1):
                        if j%10000==0:
                            print("Processed rows: "+str(j))

                        # I.e if it's time to superimpose the events. For example when creating events with 3 particles,
                        # you superimpose every chunk of three events
                        if (j-start_row_index)%i==0:
                            if j+step<=end_row_index:
                                # Putting gun events next to each other and superimposing the crystal energies for one
                                # event.
                                gun_row = flatten_and_add_zeros(get_rows_in_interval_numpy(gun_input_matrix, j, j + step),highest_number_of_particles)
                                crystal_row=crystal_energies_sigma_5_percent(add_rows_to_one(get_rows_in_interval_numpy(crystal_energy_input_matrix, j, j + step)))
                                sum_row=sum(crystal_row)

                                # Writing the created event to the corresponding files
                                f_cr.write(' '.join(str(i) for i in crystal_row))
                                f_cr.write(' \n')
                                f_gun.write(' '.join(str(i) for i in gun_row))
                                f_gun.write(' \n')
                                f_sum.write(str(sum_row))
                                f_sum.write(' \n')





if __name__ == '__main__':
    lowest_number_of_particles=int(sys.argv[1])
    highest_number_of_particles=int(sys.argv[2])
    percentage = int(sys.argv[3]) # For for example 90 %, the argument is 90 not 0.9
    crystal_energy_input_file=sys.argv[4] # Xbe file for crystal ball
    gun_input_file=sys.argv[5] # i.e. file where the data of the simulated particles are stored
    crystal_energy_output_file = sys.argv[6]
    gun_output_file = sys.argv[7]
    total_dep_energy_output_file = sys.argv[8] # sum of crystal energys. XBesum in crystal ball

    main(lowest_number_of_particles, highest_number_of_particles, percentage, crystal_energy_input_file, gun_input_file, crystal_energy_output_file,
         gun_output_file, total_dep_energy_output_file)
