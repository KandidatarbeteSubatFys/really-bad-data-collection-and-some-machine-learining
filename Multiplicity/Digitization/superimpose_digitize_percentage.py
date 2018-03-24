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


# Not used. Function used for getting multiple rows from a matrix (list format matrix, not numpy).
def get_rows_in_interval(matrix,start_row, end_row):
    out=[]
    if end_row==-1:
        end_row=len(matrix)-1
    for i in range(start_row,end_row+1):
        out.append(matrix[i])
    return out


# function used for getting multiple rows from a matrix (numpy format matrix, not list).
def get_rows_in_interval_numpy(numpy_matrix,start_row_index, end_row_index):
    if end_row_index==-1:
        end_row_index=len(numpy_matrix)-1
    out=np.zeros((end_row_index-start_row_index+1,len(numpy_matrix[0])))
    index=0
    for i in range(start_row_index,end_row_index+1):
        #print(out[index])
        #print(numpy_matrix[i])
        out[index]=numpy_matrix[i]
        index=index+1
    return out

# Not used. Functino for matrix addition
def matrix_addition(A,B):
    A=np.array(A)
    B=np.array(B)
    if len(A)==0:
        A=np.zeros((len(B),len(B[0])))
    if len(B)==0:
        B=np.zeros((len(A),len(A[0])))
    if len(A)!=len(B) or len(A[0])!=len(B[0]):
        return -1
    sum=A+B
    return sum.tolist()


# adds multiple rows of same size to one
def add_rows_to_one(matrix):
    sum_row=matrix.sum(axis=0)
    return sum_row


# Takes many singe gun-events and puts them next to each other, as well as adding the correct number of zeros
# so that every row is the same length.
def flatten_and_add_zeros(matrix,highest_number_of_particles):
    flat_numpy=matrix.flatten()
    flat_numpy=np.concatenate((flat_numpy,np.zeros(2*highest_number_of_particles-len(flat_numpy))),axis=0)
    return flat_numpy

# Unused. Same as crystal_energies_sigma_5_percent but for a matrix.
def crystal_energies_sigma_5_percent_matrix(superimposed_crystal_energy_matrix):
    out=np.zeros((len(superimposed_crystal_energy_matrix),len(superimposed_crystal_energy_matrix[0])))
    for i in range(len(superimposed_crystal_energy_matrix)):
        for j in range(len(superimposed_crystal_energy_matrix[0])):
            out[i][j]=np.random.normal(superimposed_crystal_energy_matrix[i][j],0.05*superimposed_crystal_energy_matrix[i][j],1)
    return out.tolist()


# this function gives the superimposed events some randomness. For each crystal energy, the energy is used as expected value of
# a normaldistribution and the variance is 5% of the energy.
def crystal_energies_sigma_5_percent(superimposed_crystal_energy_list):
    for i in range(len(superimposed_crystal_energy_list)):
        if superimposed_crystal_energy_list[i]!=0:
            superimposed_crystal_energy_list[i]=np.random.normal(superimposed_crystal_energy_list[i],0.05*superimposed_crystal_energy_list[i],1)
    return superimposed_crystal_energy_list


#Not used. Function for reading multiple lines from a file.
def read_line_interval(file,start_line,end_line):
    out=[]
    with open(file) as fp:
        for i, line in enumerate(fp):
            if i <= end_line and i >=start_line:
                line=line.rstrip()
                tmp_string_list = line.split(' ')
                out.append(list(map(float, tmp_string_list)))

            elif i > end_line:
                break
    return np.array(out)


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



def main(lowest_number_of_particles, highest_number_of_particles, percentage,crystal_energy_input_file, gun_input_file, output_npz_file):
    print("Time example: takes 16 minutes for 10 000 000 events for particles from 1 to 7 on pclab-232")

    print("Reading data")
    with open(crystal_energy_input_file) as f_tmp:
        for lines in f_tmp: # looks stupid and kind of is, but didn't care since it works and takes no time at all
            line = lines.rstrip()
            first_line = np.fromstring(line, dtype=np.float32, sep=' ')
            break

    number_of_crystals = len(first_line)
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

    crystal_output_matrix=np.zeros((int(events_needed_for_lowest_number_of_particles/lowest_number_of_particles*(highest_number_of_particles-lowest_number_of_particles+1)),number_of_crystals),dtype=np.float32)
    gun_output_matrix = np.zeros((int(events_needed_for_lowest_number_of_particles / lowest_number_of_particles * (highest_number_of_particles - lowest_number_of_particles + 1)), highest_number_of_particles*2), dtype=np.float32)
    total_dep_output_matrix = np.zeros((int(events_needed_for_lowest_number_of_particles / lowest_number_of_particles * (highest_number_of_particles - lowest_number_of_particles + 1)), 1), dtype=np.float32)

    # Dividing the input data so that each new set holds the single gun events needed to make the events for each number of guns
    events_for_each_gun_crystaldata={}
    events_for_each_gun_gundata = {}
    for i in range(lowest_number_of_particles,highest_number_of_particles+1):
        start_row_index = int(events_needed_for_lowest_number_of_particles / lowest_number_of_particles * sum([j for j in range(lowest_number_of_particles, i)]))
        end_row_index = int(events_needed_for_lowest_number_of_particles / lowest_number_of_particles * sum([j for j in range(lowest_number_of_particles, i + 1)])) - 1
        events_for_each_gun_crystaldata['events_for_nr_gun='+str(i)]=get_rows_in_interval_numpy(crystal_energy_input_matrix,start_row_index,end_row_index)
        events_for_each_gun_gundata['events_for_nr_gun=' + str(i)] = get_rows_in_interval_numpy(gun_input_matrix, start_row_index,end_row_index)
        print('Have extracted the events for up to gun=' + str(i) + ' of totally ' + str(highest_number_of_particles))

    # Here the superimposing is done
    index_out_row=0
    for i in range(lowest_number_of_particles,highest_number_of_particles+1):
        for j in range(int(events_needed_for_lowest_number_of_particles/lowest_number_of_particles)):
            if (index_out_row)%10000==0:
                print('Number of completed rows '+ str(index_out_row+1) + '. The total number of rows is less than '+str(int(events_needed_for_lowest_number_of_particles/lowest_number_of_particles*(highest_number_of_particles-lowest_number_of_particles+1))))
            gun_row=flatten_and_add_zeros(get_rows_in_interval_numpy(events_for_each_gun_gundata['events_for_nr_gun='+str(i)],i * j, (j+1)*i-1),highest_number_of_particles)

            crystal_row = crystal_energies_sigma_5_percent(add_rows_to_one(get_rows_in_interval_numpy(events_for_each_gun_crystaldata['events_for_nr_gun='+str(i)], i * j,(j+1)*i-1)))
            sum_row = sum(crystal_row)
            #print(gun_row)
            gun_output_matrix[index_out_row] = gun_row
            crystal_output_matrix[index_out_row] = crystal_row
            total_dep_output_matrix[index_out_row] = sum_row
            index_out_row+=1



    np.savez(output_npz_file,crystal_matrix=crystal_output_matrix,gun_matrix=gun_output_matrix,total_dep_matrix=total_dep_output_matrix)




if __name__ == '__main__':
    lowest_number_of_particles=int(sys.argv[1])
    highest_number_of_particles=int(sys.argv[2])
    percentage = int(sys.argv[3]) # For for example 90 %, the argument is 90 not 0.9
    crystal_energy_input_file=sys.argv[4] # Xbe file for crystal ball
    gun_input_file=sys.argv[5] # i.e. file where the data of the simulated particles are stored
    output_npz_file = sys.argv[6]

    main(lowest_number_of_particles, highest_number_of_particles, percentage, crystal_energy_input_file, gun_input_file, output_npz_file)