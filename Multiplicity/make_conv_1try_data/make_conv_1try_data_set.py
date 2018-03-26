import time as t
import numpy as np
import sys

def read_geom_txt():
    """"Reads the geometry file and removes white space"""
    with open('geom_xb.txt', 'r') as f:  # Open file
        lines = f.readlines()  # Reads the file to memory
    return [remove_white_space(line[8:-14].split(',')) for line in lines]


def remove_white_space(line):
    """Removes white space recursevly"""
    if isinstance(line, str):  # If we have a string, a singel entry has been past
        if line[0] == ' ':
            return remove_white_space(line[1:-1] + line[-1])  # Recursive call without the initial whitspace
        else:
            return line  # No more white space, return string
    elif isinstance(line, list):  # If a list has been past we call the function on the seperate enterys
        return [remove_white_space(entry) for entry in line]
    else:
        raise TypeError('Input most be as list or a string /Pontus')  # Unlegal type


def get_interval_of_cols(list_matrix, start_index, end_index):
    out = []
    for i in list_matrix:
        row = i
        correct_row = []
        for j in range(len(row)):
            if j > end_index:
                break
            if j >= start_index:
                correct_row.append(row[j])
        out.append(correct_row)
    return out


# intput needs to be a list matrix with integer elements in string format
def from_string_matrix_list_to_int_numpy(list_matrix):
    return np.array(list_matrix).astype(np.int)


def get_numpy_matrix_of_indexes_for_nearest_neighbours():
    return from_string_matrix_list_to_int_numpy(get_interval_of_cols(read_geom_txt(), 5, 10)) - 1


def make_conv_image(input_data):
    out = np.zeros((len(input_data), 3 * 3 * len(input_data[0])), dtype=np.float32)
    matrix_of_nearest_neighbours = get_numpy_matrix_of_indexes_for_nearest_neighbours()
    for i in range(len(input_data)):
        if i % 1000 == 0:
            print("Processed rows when making convolution data: " + str(i))
        tmp = np.array([[], [], []])
        for j in range(len(input_data[0])):
            tmp2 = np.zeros(3 * 3, dtype=np.float32)
            index = 0
            for k in matrix_of_nearest_neighbours[j]:
                if k == -1:
                    tmp2[index] = 0
                else:
                    tmp2[index] = input_data[i, k]
                index = index + 1
            tmp2[index] = input_data[i, j]
            tmp2 = tmp2.reshape(3, 3)
            tmp = np.concatenate((tmp, tmp2), axis=1)
        out[i] = np.concatenate((tmp[0], tmp[1], tmp[2]), axis=0)
    input_data = []
    return out


def main(ord_mult_data_set_npz,conv_data_set_npz):

    ord_data_set=np.load(ord_mult_data_set_npz)
    x_batch_eval = make_conv_image(ord_data_set['x_batch_eval'])
    x_batch_train = make_conv_image(ord_data_set['x_batch_train'])
    np.savez(conv_data_set_npz,x_batch_train=x_batch_train,y_batch_train=ord_data_set['y_batch_train'],x_batch_eval=x_batch_eval,y_batch_eval=ord_data_set['y_batch_eval'])




if __name__ == '__main__':
    ord_mult_data_set_npz=sys.argv[1]
    conv_data_set_npz=sys.argv[2]
    main(ord_mult_data_set_npz,conv_data_set_npz)






