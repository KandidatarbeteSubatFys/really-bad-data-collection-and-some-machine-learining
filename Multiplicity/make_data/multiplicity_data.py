import time as t
import numpy as np
import sys



# IMPORTANT: the input data must be ordered from the smallest number of guns to the largest number of guns.
# Also important: use approximately the same number of events per each number of guns used.


#This functions is used to get just the energy values and not the cos(theta) values from the gun matrix.
# So it basicly removes every odd column of the gun matrix. This has no real use for the multiplicity data, but
#I used it anyway because maybe I need it later on.
def just_energies_in_gun(y_batch,number_particles):
    tmp=y_batch[:,0]
    for i in range(number_particles-1):
        tmp=np.c_[tmp,y_batch[:,3*i+3]]

    return tmp


# Used to get the last index where the (number of guns)=(number of particles)-1.
def get_last_index_before_highest_number_of_guns(y_batch,number_particles):
    for i in range(y_batch.shape[0]):
        tmp = np.trim_zeros(y_batch[i])
        if tmp.shape[0] == number_particles:
            return i - 1
    return -1


# Takes in two numpy arrays and shuffles them by their rows the same way
def shuffle_two_numpy_arrays_the_same_way_by_rows(x_batch,y_batch):
    if len(x_batch)!=len(y_batch):
        return -1, -1
    matrix_to_shuffle = np.c_[x_batch, y_batch]
    np.random.shuffle(matrix_to_shuffle)
    return matrix_to_shuffle[:, [i for i in range(len(x_batch[0]))]], matrix_to_shuffle[:, [i for i in range(len(x_batch[0]), len(x_batch[0]) + len(y_batch[0]))]]


# Returns an interval of rows for a numpy matrix. In for example matlab you can just write matrix(start_row:end_row,:)
# but I couldn't find the equivalent for that in python.
def get_rows_numpy_array(numpy_array,index_first_row, index_last_row):
    if index_last_row==-1:
        index_last_row=len(numpy_array)-1
    return numpy_array[[i for i in range(index_first_row,index_last_row+1)],:]


# Converts the gun-matrix converted into matrix of one-hot vectors. The output has the list format, i.e. isn't numpy anymore.
# Perhaps a bit inefficient to convert to string and then back to float, but this method works nevertheless.
def get_one_hot_numpy_matrix_from_numpy_array(y_batch,number_particles):
    y_batch_without_zeros=np.zeros(len(y_batch),dtype=np.int)
    for i in range(len(y_batch)):
        y_batch_without_zeros[i]=len(np.trim_zeros(y_batch[i]))
    y_batch_out=np.zeros((len(y_batch_without_zeros),number_particles+1),dtype=np.float32)

    index=0
    for i in y_batch_without_zeros:
        row_str = "0 " * i + str(1) + " " + (number_particles - i) * "0 "
        row_str = row_str.rstrip()
        tmp_string_list = row_str.split(' ')
        y_batch_out[index]=np.array(tmp_string_list,dtype=np.int)
        index+=1
    return y_batch_out



def main(npz_input_file,npz_output_file):


    # First the data from
    # the textfiles that came from the simulation is converted into numpy arrays. Then the cos(theta) values from the
    # gun matrix is removed (I could have just extracted the number of guns imidiately, but I wanted to keep the code
    # slightly more general). Then, since we don't want to evaluate on the highest number of guns that was used (for
    # example if you have data for up to 7 guns, then you will train on all the data, but only evaluate on 6) all
    # the other data is extracted. Then this extracted data is divided into a training batch and an evaluation batch.
    #
    # The training data from the extracted data is then merged with the highest-gun data. Data for cases where nothing
    # happends is also added to the training data (gun energy=0, deposited energy=0). Then the data is shuffled (but the
    # corresponing pair of input data and output data is always on the same "row"). Lastly the output data on turned
    # into one-hot vectors because its a good representation of the number of guns used for the loss function.

    print('Reading data')

    data_set=np.load(npz_input_file)
    print("Reading input data")
    x_batch =data_set["crystal_matrix"]
    print("Reading output data")
    y_batch = data_set["gun_matrix"]

    number_of_events=len(y_batch)
    print("Number of events: "+str(number_of_events))
    number_particles=int(len(y_batch[0])/2)

    print("Creation of training and evaluation batches")

    y_batch=just_energies_in_gun(y_batch,number_particles)

    # Here the index for when the highest-gun data starts is extracted (the data file must be ordered from lowest amount
    # of guns to highest for this method to work).
    last_index_before_highest_number_of_guns=get_last_index_before_highest_number_of_guns(y_batch,number_particles)
    if last_index_before_highest_number_of_guns==-1:
        print(
            "Error: No index for last evaluation data found. Check if number_particles is correct or if the data really is ordered from lowest nubmer of guns to highest.")
        return

    # All the data except for the data for the highest amount of guns is extracted
    x_batch_highest_gun_removed=get_rows_numpy_array(x_batch, 0, last_index_before_highest_number_of_guns)
    y_batch_highest_gun_removed = get_rows_numpy_array(y_batch, 0, last_index_before_highest_number_of_guns)

    # The extracted data is then shuffled
    x_batch_highest_gun_removed, y_batch_highest_gun_removed = shuffle_two_numpy_arrays_the_same_way_by_rows(x_batch_highest_gun_removed,y_batch_highest_gun_removed)
    if isinstance(x_batch_highest_gun_removed,int):
        print("Error when shuffling: x_batch and y_batch not the same length")
        return

    # The shuffled extracted data is divided in a training and a evaluation set
    x_batch_eval = get_rows_numpy_array(x_batch_highest_gun_removed, 0, int(0.2*len(x_batch_highest_gun_removed))-1)
    y_batch_eval = get_rows_numpy_array(y_batch_highest_gun_removed, 0, int(0.2 * len(y_batch_highest_gun_removed))-1)
    x_batch_train_without_highest_gun_and_zeros = get_rows_numpy_array(x_batch_highest_gun_removed,
                                                                       int(0.2 * len(x_batch_highest_gun_removed)), -1)
    y_batch_train_without_highest_gun_and_zeros = get_rows_numpy_array(y_batch_highest_gun_removed,
                                                                       int(0.2 * len(y_batch_highest_gun_removed)), -1)

    # The training set of the extracted data is merged with the highest-number-of-guns data, as well as with zero events
    x_zeros=np.zeros((int(len(x_batch_train_without_highest_gun_and_zeros)/(number_particles-1)),162))
    y_zeros = np.zeros((int(len(y_batch_train_without_highest_gun_and_zeros) / (number_particles - 1)), number_particles))

    x_batch_highest_number_of_guns=get_rows_numpy_array(x_batch,last_index_before_highest_number_of_guns+1,-1)
    y_batch_highest_number_of_guns = get_rows_numpy_array(y_batch, last_index_before_highest_number_of_guns + 1, -1)
    x_batch_highest_number_of_guns_decreased = get_rows_numpy_array(x_batch_highest_number_of_guns,0,len(x_zeros)-1)
    y_batch_highest_number_of_guns_decreased = get_rows_numpy_array(y_batch_highest_number_of_guns, 0, len(y_zeros) - 1)

    x_batch_train=np.r_[np.r_[x_zeros,x_batch_highest_number_of_guns_decreased],x_batch_train_without_highest_gun_and_zeros]
    y_batch_train = np.r_[np.r_[y_zeros, y_batch_highest_number_of_guns_decreased],y_batch_train_without_highest_gun_and_zeros]

    # The training data is shuffled
    x_batch_train, y_batch_train = shuffle_two_numpy_arrays_the_same_way_by_rows(x_batch_train,y_batch_train)

    # The above was more general than what would be needed to determine the multiplicity. But the following is more specific:

    # the evaluation input data is turned from a numpy array to a list
    # The number of guns matrix is now converted into a matrix of one-hot vectors. Each row representing the number of guns for that event.
    y_batch_eval=get_one_hot_numpy_matrix_from_numpy_array(y_batch_eval,number_particles)

    # same procedure for the training data
    y_batch_train = get_one_hot_numpy_matrix_from_numpy_array(y_batch_train,number_particles)


    np.savez(npz_output_file,y_batch_train=np.array(y_batch_train),x_batch_train=x_batch_train,y_batch_eval=np.array(y_batch_eval),x_batch_eval=x_batch_eval)



if __name__ == '__main__':
    npz_input_file = sys.argv[1]
    npz_output_file=sys.argv[2]
    main(npz_input_file,npz_output_file)






