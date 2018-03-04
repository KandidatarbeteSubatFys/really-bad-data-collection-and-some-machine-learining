import numpy as np
import matplotlib.pyplot as plt


def read_geom_txt():
    """"Reads the geometry file and removes white space"""
    with open('geom_xb.txt', 'r') as f:         # Open file
        lines = f.readlines()                   # Reads the file to memory
    return [remove_white_space(line[8:-14].split(',')) for line in lines]

def remove_white_space(line):
    """Removes white space recursevly"""
    if isinstance(line, str):                                       # If we have a string, a singel entry has been past
        if line[0] == ' ':
            return remove_white_space(line[1:-1] + line[-1])        # Recursive call without the initial whitspace
        else:
            return line                                             # No more white space, return string
    elif isinstance(line, list):                                    # If a list has been past we call the function on the seperate enterys
        return [remove_white_space(entry) for entry in line]
    else:
        raise TypeError('Input most be as list or a string /Pontus')    # Unlegal type


def string_matrix_to_numeric(line_list):
    """ Cast the strings to there useful type"""
    num_matrix = []
    for line in line_list:
        temp = []
        temp.append(int(line[0]))       # int index
        temp.append(line[1])            # str type
        temp.append(float(line[2]))     # float theta
        temp.append(float(line[3]))     # float phi
        temp.append(float(line[4]))     # float psi
        temp.append(int(line[5]))       # int nabers
        temp.append(int(line[6]))
        temp.append(int(line[7]))
        temp.append(int(line[8]))
        temp.append(int(line[9]))
        temp.append(int(line[10]))
        num_matrix.append(temp)
    return num_matrix


def get_theta(line):
    """ Returns theta of a single line of numeric matrix or a list of them """
    if isinstance(line[0], int):            # Acts on a singel entry
        return line[2]
    elif isinstance(line[0], list):         # Acts on multiple lines
        return [i[2] for i in line]


def get_phi(line):
    """ Returns phi of a single line of numeric matrix or a list of them """
    if isinstance(line[0], int):            # Acts on a singel entry
        return line[3]
    elif isinstance(line[0], list):         # Acts on a multiple entry
        return [i[3] for i in line]


def remove_multipul_copies(angle_list):     # Returns a sorted list there all copys has been removed
    if len(angle_list) == 1:                # If the list is just a single element return it
        return angle_list
    sorted_list = sorted(angle_list)                                            # Sort the original list
    new_list = [sorted_list[0]]
    for angel_index in range(1, len(angle_list)):
        if not sorted_list[angel_index] == sorted_list[angel_index -1]:    # If the element are not equal add to list
            new_list.append(sorted_list[angel_index])
    return new_list


def index_matrix_2D(num_matrix):
    """Generates a matrix with indecis, coresponding to where in a 2D image the cristalls in xb should be
    mapped. The mapping is such that all cristalls is is orderd in incressing theta columnwise och the
    orderd according to phi within the column. """
    thetas = remove_multipul_copies(get_theta(num_matrix))
    index_list_for_thetas = []                                                  # get the different thetas
    for theta in thetas:
        temp = []
        for line in num_matrix:
            if get_theta(line) == theta:                                        # The theta we locking for, append
                temp.append(line[0]-1)
        index_list_for_thetas.append(temp)      # A list of list contaning the index of all the index correspunding to a theta

    def sub_lines(indicies):                                                    # Gives a sub matrix of the input num_matrix
        return [num_matrix[i] for i in indicies]

    num_theta = len(index_list_for_thetas)                                      # Number of diffrent thetas
    num_phi = max([len(sub_list) for sub_list in index_list_for_thetas])        # max number of phi for a given theta
    out = []

    for index_list in index_list_for_thetas:                            # For all thetas sort according to phi
        num_sub_matrix = sub_lines(index_list)
        phis = sorted(get_phi(num_sub_matrix))                          # Sort after phi for a given theta
        temp = []
        for phi in phis:
            for line in num_sub_matrix:
                if get_phi(line) == phi:                                # The theta we are looking for, append
                    temp.append(line[0])
        out.append(temp)
    return out, num_theta, num_phi


def generate_2D_index_matrix():                                         # Calls index_matrix_2D with correct inputs
    return index_matrix_2D(string_matrix_to_numeric(read_geom_txt()))


def index_matrix_2D_phi(num_matrix):
    """ Order the 'pixels' in theta and phi but now the ordering is faithful in phi as well. So for every given theta
    and phi in xb there is a spot in the matrix, results in a large amount of dead pixels but describe the spacial
    relations in a more complete way. """
    thetas = remove_multipul_copies(get_theta(num_matrix))
    phis = remove_multipul_copies(get_phi(num_matrix))
    out = [[0 for i in range(len(phis))] for i in range(len(thetas))]

    def index_of_theta_phi(line):            # Returns the index of the lines theta and phi in the thetas and phis list
        return thetas.index(get_theta(line)), phis.index(get_phi(line))

    for line in num_matrix:                                 # Looping over lines and fill the corresponding pixels
        index_theta, index_phi = index_of_theta_phi(line)
        out[index_theta][index_phi] = line[0]

    num_theta = len(thetas)
    num_phi = len(phis)
    return out, num_theta, num_phi


def generate_2D_index_matrix_phi():                                         # Calls index_matrix_2D_phi with the
    return index_matrix_2D_phi(string_matrix_to_numeric(read_geom_txt()))   # correct inputs


def map_2D_picture(energy_list, index_matrix_2D, num_theta, num_phi):   # Given a index matrix map the energy accordingly
    map_2D = np.zeros([num_theta, num_phi])
    for i in range(num_theta):                                          # Looping over ''picture''
        for j in range(len(index_matrix_2D[i])):
            if index_matrix_2D[i][j] == 0:                              # list[-1] = last element in list
                continue                                                # Skips this iteration of the loop
            map_2D[i][j] = energy_list[index_matrix_2D[i][j] - 1]

    return map_2D


def plot_2D_map(map_2D):
    """Plots a images of a single event in normalized units. """
    plt.imshow(map_2D / np.sum(map_2D))                             # Normalized plot
    plt.colorbar()
    plt.title('Total energy: ' + str(np.sum(map_2D)) + ' MeV')
    plt.show()


if __name__ == '__main__':
    # Just for test, the file should just be imported, not meant to be run.

    energy_string = '0.000000 0.000000 0.000000 0.000000 0.000000 2.454613 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.571859 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ' \
                    '0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000'
    energy_list = [float(i) for i in energy_string.split(' ')]
    index_matrix, num_theta, num_phi = generate_2D_index_matrix_phi()
    map_2D = map_2D_picture(energy_list, index_matrix, num_theta, num_phi)
    plot_2D_map(map_2D)
