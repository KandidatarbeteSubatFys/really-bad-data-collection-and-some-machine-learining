""" Some methods concerning loss functions, define networks and training in tensorflow.
    -- Spring 2018; ce2018
"""
import tensorflow as tf
import numpy as np
import itertools as it


def percent_error(x, y):
    """ Obsolete! Constructs a network to compute a relative error.
    Args: x: tensor predicted value.
          y: tensor correct value.
    Out:  unnamed: tensor the relative error.
    """
    temp1, temp2 = tf.split(y, 2, axis=1)                                   # Split the angle and energy values.
    E_ = tf.concat([temp1, tf.ones(tf.shape(temp1), dtype=tf.float32)], 1)  # Fill out the energy value with ones
    return tf.divide(x-y, E_)                                               # The relative error, divides angel with one


def hidden_laysers(input, dim, nr, relu=True, dtype=tf.float32):
    """Constructs a number of hidden layer in our graf. The layer is a fc layer ontop of the input. Next layers is of
    the same dimension as the input. The function works recursively. Example of use in def_fc_layers.
    Args:   input: tensor the tensor that should be built ontop of.
            dim: int the with of the input and the output.
            nr : int the number of layers ontop of input.
            relu: boolean should activation in form of relu be used.
            dtype: tf.type the type of the data in the network.
    Out:    unnamed: tensor the graf with the hidden layers.
    """
    W = tf.Variable(tf.truncated_normal([dim, dim], stddev=0.1), dtype=dtype)           # Def weights of a single layer.
    b = tf.Variable(tf.zeros([dim]), dtype=dtype)                                       # Def baias for a single layer.
    if  nr < 0:
        raise ValueError('Number of laysers most be a positive integer /Pontus')
    elif nr == 0:                                                                       # No layer should be added.
        return input
    elif nr == 1:                                                                       # End of recursion. Return input
        if relu:                                                                        # with a layer on top.
            return tf.nn.relu(tf.matmul(input, W) + b)
        else:
            return tf.matmul(input, W) + b
    else:                                                               # Adds a layer and call the
        if relu:                                                        # Function again with a lower number of layers.
            return tf.nn.relu(tf.matmul(hidden_laysers(input, dim, nr-1, relu=relu, dtype=dtype), W) + b )
        else:
            return tf.matmul(hidden_laysers(input, dim, nr-1, relu=relu, dtype=dtype), W) + b


def def_fc_layers(input, start_nodes, end_nodes, hidden_nodes, nr_hidden_laysers, relu=True, dtyp=tf.float32):
    """Defines a fc graf with help of the function hidden layers. The hidden layers are of the same size and the
    first and last set of weight make the dimension fit the input size and the output size.
    Args:   input: tensor the tensor the network should be built on.
            start_nodes: int the number of nodes in the input layer.
            end_nodes: int the number of nodes in the output layer.
            hidden_nodes: int the number of nodes in the hidden layer.
            relu: boolean should the activation relu be used on the layers.
            dtype: tf.type the data type in the graf.
    Out:    unnamed: tensor graf built ontop of the input.
    """
    W_start = tf.Variable(tf.truncated_normal([start_nodes, hidden_nodes], stddev=0.1), dtype=dtyp) # Start weights
    b_start = tf.Variable(tf.zeros([hidden_nodes]))                                                 # start baias
    W_end = tf.Variable(tf.truncated_normal([hidden_nodes, end_nodes], stddev=0.1), dtype=dtyp)     # End weights
    b_end = tf.Variable(tf.zeros([end_nodes]))                                                      # End baias

    if relu:
        temp = tf.nn.relu(tf.matmul(input, W_start) + b_start)                                  # Put the fist layer on.
        temp2 = hidden_laysers(temp, hidden_nodes, nr_hidden_laysers-1, relu=relu, dtype=dtyp)  # Adds the hidden layers
        return tf.matmul(temp2, W_end) + b_end                                                  # Makes the output layer
    else:
        temp = tf.matmul(input, W_start) + b_start
        temp2 = hidden_laysers(temp, hidden_nodes, nr_hidden_laysers-1, relu=relu, dtype=dtyp)
        return tf.matmul(temp2, W_end) + b_end


def def_loss(y, y_, name_of_loss, num_splits=None):
    """A way to store old loss functions and in the same time still have them accessible, just call this function with
    the input y to the loss, the 'correct' result y_ and the name of the loss function to use.
    Args:   y: tensor predicted value from network.
            y_: tensor the correct value.
    Additional arguments can be needed for a specific loss function.
    Out:    unnamed: tensor a loss function.
    """
    if name_of_loss == 'kombinations_two':
        # Energy predict of two gammas, looks at the two cases.
        return tf.reduce_mean(tf.minimum(tf.reduce_sum(tf.square(y-y_), 1),
                                         tf.reduce_sum(tf.square(tf.reverse(y, [-1])-y_), 1)))
    elif name_of_loss == 'kombinationer':
        # Energy values and cos(theta) values, squar loss function, looks att all combinations.
        if num_splits == None:
            raise ValueError('This loss functions requires a parameter num_splits equal to the (max) number of particles')
        return split_energy_angle_comb(y, y_, num_splits)
    else:
        raise ValueError('The name of the loss function is not recognised')


def def_architecture(x, name_of_architectures, input_size=None, output_size=None, hidden_layers=None, hidden_nodes=None):
    """A way to store old architectures in the code, just call the function with input and name.
    Args:   x: tensor input tensor.
            name_of_architecture: str the name of the architecture that should be used.
    Additional argument may be recurred for a specific architecture.
    Out:    unnamed: tensor the network.
    """
    if name_of_architectures == 'same_size_hidden layers':
        # fc layers with same size, except input and output layers.
        if hidden_layers == None or hidden_nodes == None or input_size == None or output_size == None:
            raise ValueError('Not all input parameters is defined /Pontus')
        return def_fc_layers(x, input_size, output_size, hidden_nodes, hidden_laysers)
    else:
        raise ValueError('Name of arcitecture is not recognized')


def split_energy_angle_comb(y, y_, num_splits, lam=1, tree=True, off_set=0):
    """Loss function used when the network predict energy's and cos(theta) of an arbitrary number of particles. The
    energy is normalized and summed with the angle quadratic. It looks at all combination so se the right match
    between the predicted and the goal values. The energy's and angel is always acossieated so just the combination
    of the energy angle block.
    Args:   y: tensor predicted values alternating energy and angel data.
            y_: tensor the 'correct' y.
            num_split: int the number of 'blocks', the number of particles.
            lam: float a weigting between the energy and angle in the loss function.
            tree: boolean if true, the minimum operator will be used in a tree structure
                          if false, the minimum operator will just be taken iterative.
    Out:    loss: tensor a network for calculating the loss.
    """
    splited_y = tf.split(y, num_splits, axis=1)                                         # Split y in particle blocks
    splited_y_ = tf.split(y_, num_splits, axis=1)                                       # Split y_ in particle blocks
    temp_shape = tf.shape(tf.split(splited_y[0], 2, axis=1))                            # Shape of batch                    ToDo: look att this, should maybe be [0] after split.

    def one_comb_loss(splited_y, splited_y_, index_list):                               # Nested funk to calc loss for a
        temp = tf.zeros(temp_shape, dtype=tf.float32)                                   # given permutation, index_list.
        for i in range(len(index_list)):                                                # loop over particle's.
            E, cos = tf.split(splited_y[i], 2, axis=1)                                  # Split energy and angel
            E_, cos_ = tf.split(splited_y_[index_list[i]], 2, axis=1)                   # -II-
            temp = temp + lam*tf.square(tf.divide(E-E_, E_+off_set)) + tf.square(cos - cos_)    # Add loss iterative
        return temp                                                                     # Return the loss

    def minimize_step(tensor_list):                 # Nested funk, one step in the tree structure.
        if len(tensor_list) % 2 == 0:               # Even number of losses, find minimum of between pairs.
            return [tf.minimum(tensor_list[i], tensor_list[i+1]) for i in range(0, len(tensor_list), 2)]
        else:                                       # Odd number of losses, find min between pairs, last unpaired.
            new_list_of_tensors = [tf.minimum(tensor_list[i], tensor_list[i+1]) for i in range(0, len(tensor_list)-1, 2)]
            new_list_of_tensors.append(tensor_list[-1])     # Append the unpaired loss.
            return new_list_of_tensors

    # All losses in a list. Inner loop over all permutations.
    list_of_tensors = [one_comb_loss(splited_y, splited_y_, index_list) for index_list in it.permutations(range(num_splits), num_splits)]

    if tree == False:
        loss = tf.divide(tf.constant(1, dtype=tf.float32), tf.zeros(temp_shape, dtype=tf.float32))  # Init inf loss
        for i in range(len(list_of_tensors)):
            loss = tf.minimum(loss, list_of_tensors[i])                     # Successive min, with previous best.
    else:                                                                   # Tree structure.
        while len(list_of_tensors) > 1:                                     # while nr loss > 1.
            list_of_tensors = minimize_step(list_of_tensors)                # Successive min, returns a smaller list.
        loss = list_of_tensors[0]                                           # Bind best to loss name.

    return tf.reduce_mean(loss)                                             # Returns the men over the batch.


def min_sqare_loss_combination(y, y_, lam=1, off_set=0):
    """Equivalent split_energy_angle_comb but act on list insted of tensors. A single evens should just be given. Is
    used to match predicted and correct data for data analysis.
    Args:   y: list predicted.
            y_: list correct.
            lam: float weight between angle and energy.
    Out: index_min: list of the index comb that had min loss, the right matching.
    Note: A single event should be given, not a batch!
    """
    if not len(y) == len(y_):
        raise ValueError('y and y_ most be of same length')

    index_min = False
    temp_min = np.inf                                                           # Init min loss to inf.
    for index_list in it.permutations(range(int(len(y)/2)), int(len(y)/2)):     # Looping over all permutations.
        temp = 0
        for i in range(int(len(y)/2)):                                          # Successively adding up the loss.
            temp = temp + lam*np.power((y[2*i] - y_[2*index_list[i]])/(y_[2*index_list[i]] + off_set), 2) + np.power(y[2*i+1] - y_[2*index_list[i]+1], 2)

        if temp < temp_min:                                 # If loss is less the min, save loss and the permutation.
            temp_min = temp
            index_min = index_list

    if index_min == False:
        raise ValueError('Something is horribly wrong /Pontus ')

    return index_min                                        # Return the permutation with min loss.


def split_energy_angle_angle_comb(y, y_, num_splits, lam=1, lam2=1, tree=True, off_set=0):
    """Loss function used when the network predict energy's, cos(theta) and phi of an arbitrary number of particles. The
    energy is normalized and summed with the angles quadratic. It looks at all combination so se the right match
    between the predicted and the goal values. The energy's and angel is always acossieated so just the combination
    of the energy angle block.
    Args:   y: tensor predicted values alternating energy and angel data.
            y_: tensor the 'correct' y.
            num_split: int the number of 'blocks', the number of particles.
            lam: float a weigting between the energy and angle in the loss function.
            tree: boolean if true, the minimum operator will be used in a tree structure
                          if false, the minimum operator will just be taken iterative.
    Out:    loss: tensor a network for calculating the loss.
    """
    splited_y = tf.split(y, num_splits, axis=1)                                         # Split y in particle blocks
    splited_y_ = tf.split(y_, num_splits, axis=1)                                       # Split y_ in particle blocks
    temp_shape = tf.shape(tf.split(splited_y[0], 3, axis=1)[0])                         # Shape of batch                    ToDo: look att this, should maybe be [0] after split.

    def one_comb_loss(index_list):                                                      # Nested funk to calc loss for a
        temp = tf.zeros(temp_shape, dtype=tf.float32)                                   # given permutation, index_list.
        for i in range(len(index_list)):                                                # loop over particle's.
            E, cos, phi = tf.split(splited_y[i], 3, axis=1)                             # Split energy, angle and angle
            E_, cos_, phi_ = tf.split(splited_y_[index_list[i]], 3, axis=1)             # -II-
            temp = temp + lam * tf.square(tf.divide(E - E_, E_ + off_set)) + tf.square(cos - cos_) + lam2*tf.square(tf.mod(phi - (phi_+np.pi) + np.pi, 2*np.pi) - np.pi)/(2*np.pi*1000)  # Add loss iterative
        return temp                                                                     # Return the loss

    def minimize_step(tensor_list):                 # Nested funk, one step in the tree structure.
        if len(tensor_list) % 2 == 0:               # Even number of losses, find minimum of between pairs.
            return [tf.minimum(tensor_list[i], tensor_list[i+1]) for i in range(0, len(tensor_list), 2)]
        else:                                       # Odd number of losses, find min between pairs, last unpaired.
            new_list_of_tensors = [tf.minimum(tensor_list[i], tensor_list[i+1]) for i in range(0, len(tensor_list)-1, 2)]
            new_list_of_tensors.append(tensor_list[-1])     # Append the unpaired loss.
            return new_list_of_tensors

    # All losses in a list. Inner loop over all permutations.
    list_of_tensors = [one_comb_loss(index_list) for index_list in it.permutations(range(num_splits), num_splits)]

    if tree == False:
        loss = tf.divide(tf.constant(1, dtype=tf.float32), tf.zeros(temp_shape, dtype=tf.float32))  # Init inf loss
        for i in range(len(list_of_tensors)):
            loss = tf.minimum(loss, list_of_tensors[i])                     # Successive min, with previous best.
    else:                                                                   # Tree structure.
        while len(list_of_tensors) > 1:                                     # while nr loss > 1.
            list_of_tensors = minimize_step(list_of_tensors)                # Successive min, returns a smaller list.
        loss = list_of_tensors[0]                                           # Bind best to loss name.

    return tf.reduce_mean(loss)                                             # Returns the mean over the batch.


def min_sqare_loss_combination_phi(y, y_, lam=1, lam2=1, off_set=0):
    """Equivalent to split_energy_angle_angle_comb but act on a list insted of tensors. A single evens should just be given. Is
    used to match predicted and correct data for data analysis.
    Args:   y: list predicted.
            y_: list correct.
            lam: float weight between angle and energy.
    Out: index_min: list of the index comb that had min loss, the right matching.
    Note: A single event should be given, not a batch!
    """
    if not len(y) == len(y_):
        raise ValueError('y and y_ most be of same length')

    index_min = False
    temp_min = np.inf                                                           # Init min loss to inf.
    for index_list in it.permutations(range(int(len(y)/3)), int(len(y)/3)):     # Looping over all permutations.
        temp = 0
        for i in range(int(len(index_list)/3)):                                          # Successively adding up the loss.
            temp = temp + lam*np.power((y[3*i] - y_[3*index_list[i]])/(y_[3*index_list[i]] + off_set), 2) +\
                   np.power(y[3*i+1] - y_[3*index_list[i]+1], 2) + lam2*np.power(np.mod(y[3*i+2]-(y[3*index_list[i]+2]+np.pi) + np.pi, 2*np.pi) - np.pi, 2)/(2*np.pi)

        if temp < temp_min:                                 # If loss is less the min, save loss and the permutation.
            temp_min = temp
            index_min = index_list

    if index_min == False:
        raise ValueError('Something is horribly wrong /Pontus ')

    return index_min                                        # Return the permutation with min loss.


def multiplicitet(x, max_number_of_particles, threshold=0.05):
    """Network to calculate multiplicity, by looking after energy's below a given threshold. Works on alternating energy
    and angle tensors.
    Args:   x: tensor the input that the multiplicity should be calculated.
            max_number_of_particles: int the number of maximum particles, maximum multiplicity.
            threshold: float the threshold energy.
    Out:    unnamed: tensor network for calc multiplicity.
    """
    splited_x = tf.split(x, 2*max_number_of_particles, axis=1)      # Split all energys and all angles.
    temp_list = []
    for i in range(max_number_of_particles):                        # Pick out energys.
        temp_list.append(splited_x[2*i])

    E = tf.concat(temp_list, axis=1)                                # All energys in a tensor.
    comp = threshold*tf.ones(tf.shape(E), dtype=tf.float32)         # The tensor E should be comped to.

    # Comper -> boolean, cast to float false -> 0, true -> 1, sum to get multiplicity.
    return tf.reduce_sum(tf.cast(tf.less(comp, E), tf.float32), axis=1)


def accuracy(y, y_, number_of_partikles, threshold=0.05):
    """Network to calc accuracy in multiplicity. Compeers if multiplicity gives the same answer for y and y_. See
    multiplicity for ues of threshold.
    Args:   y: tensor predicted energy and theta tensor.
            y_: tensor correct energy and theta tensor.
            threshold: float see multiplicity.
    """
    # Equal multiplicity -> boolean, cast tp float, mean over batch to get accuracy.
    return tf.reduce_mean(tf.cast(tf.equal(multiplicitet(y, number_of_partikles, threshold=threshold),
                                           multiplicitet(y_, number_of_partikles, threshold=threshold)), dtype=tf.float32))


def get_nr_parameters():
    """Calculate the number of (trainable) parameters that has been initiated. Note, often tensors are initiated
    globally and may be stored between runs in a parametersweep, in that case all previous trainable varibles will also
    be seen by the function.
    Out:    unnamed: float number of trainable variables.
    """
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


if __name__ == '__main__':
    print('Some methods concerning loss functions, define networks and training in tensorflow.')
    print('--Spring 2018; ce2018')

