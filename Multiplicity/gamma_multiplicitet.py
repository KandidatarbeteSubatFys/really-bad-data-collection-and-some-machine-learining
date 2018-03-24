import tensorflow as tf
import time as t
import numpy as np
import matplotlib.pyplot as plt


# IMPORTANT: the input data must be ordered from the smallest number of guns to the largest number of guns.
# Also important: use approximately the same number of events per each number of guns used.


# function for reading file of floats with spaces between them. Not the same as Pontus' function, mine might be a little
# slower but I made a new because I thought that Pontus' function was causing problems for me, but it was probably
# something else. Anyway, it works and I havn't bothered switching.
def read_data(file):
    out=[]
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            tmp_string_list = line.split(' ')
            out.append(list(map(float, tmp_string_list)))

    return out

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

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# This function randomly selects some rows from the matrixes batch_x and batch_y and returns the rows as two matrixes.
# The number of rows is determined by the batch_size variable. This is exacly the same function that Pontus wrote.
def gen_sub_set(batch_size, batch_x, batch_y):
    if not len(batch_x) == len(batch_y):
        raise ValueError('Lists most be of same length /Pontus')
    index_list = np.random.randint(0, len(batch_x), size=batch_size)
    return [batch_x[index].tolist() for index in index_list], [batch_y[index].tolist() for index in index_list]


#This functions is used to get just the energy values and not the cos(theta) values from the gun matrix.
# So it basicly removes every odd column of the gun matrix. This has no real use for the multiplicity data, but
#I used it anyway because maybe I need it later on.
def just_energies_in_gun(y_batch,number_particles):
    tmp=y_batch[:,0]
    for i in range(number_particles-1):
        tmp=np.c_[tmp,y_batch[:,2*i+2]]

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
def get_one_hot_list_matrix_from_numpy_array(y_batch,number_particles):
    y_batch_without_zeros=[len(np.trim_zeros(row)) for row in y_batch]
    y_batch_out=[]
    for i in y_batch_without_zeros:
        row_str = "0 " * i + str(1) + " " + (number_particles - i) * "0 "
        row_str = row_str.rstrip()
        tmp_string_list = row_str.split(' ')
        y_batch_out.append(list(map(float, tmp_string_list)))
    return y_batch_out


def get_y_for_specified_layers_and_nodes(x,number_of_hidden_layers,number_of_nodes_per_hidden_layer,number_particles):
    if number_of_hidden_layers==0:
        return -1
    weights={}
    biases={}
    weights["W" + str(1)] = tf.Variable(tf.truncated_normal([162, number_of_nodes_per_hidden_layer], stddev=0.1), dtype=tf.float32)
    biases["b" + str(1)] = tf.Variable(tf.ones([number_of_nodes_per_hidden_layer]), dtype=tf.float32)
    for i in range(1,number_of_hidden_layers):
        weights["W"+str(i+1)]=tf.Variable(tf.truncated_normal([number_of_nodes_per_hidden_layer, number_of_nodes_per_hidden_layer], stddev=0.1), dtype=tf.float32)
        biases["b"+str(i+1)]=tf.Variable(tf.ones([number_of_nodes_per_hidden_layer]), dtype=tf.float32)
    weights["W" + str(number_of_hidden_layers+1)] = tf.Variable(tf.truncated_normal([number_of_nodes_per_hidden_layer, number_particles + 1], stddev=0.1),dtype=tf.float32)
    biases["b" + str(number_of_hidden_layers+1)] = tf.Variable(tf.ones([number_particles + 1]), dtype=tf.float32)
    y = x
    for i in range(number_of_hidden_layers):
        y=tf.nn.relu(tf.matmul(y, weights["W"+str(i+1)]) + biases["b"+str(i+1)])
    y=tf.matmul(y,weights["W"+str(number_of_hidden_layers+1)]) + biases["b"+str(number_of_hidden_layers+1)]
    return y

def main(npz_file,number_particles,number_of_hidden_layers,number_of_nodes_per_hidden_layer):
    print('Initializing variables')

    # Making placeholders for the inputdata (x) and the correct output data (y_)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 162]) #162=number of crystals in crystal ball detector
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, number_particles+1])

    y=get_y_for_specified_layers_and_nodes(x,number_of_hidden_layers,number_of_nodes_per_hidden_layer,number_particles)
    if isinstance(y,int):
        print("Error: number of hidden layers need to be at least one")
        return


    # As the loss funtion the softmax-crossentropy is used since it's common for classification problems.
    # To optimize the variables, Adam Optimizer is used since it fairs well in comparisons and is easy to use.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # To check the accuracy, the highest argument of the outputlayer and the one-hot-vector (one-hot is just a way to
    # represent the correct number of guns) is compared

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # converts boelean to ones and zeros and takes the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # Reading the data set from the npz file

    print('Reading data')
    data_set=np.load(npz_file)
    x_batch_train=data_set['x_batch_train']
    y_batch_train = data_set['y_batch_train']
    x_batch_eval = data_set['x_batch_eval']
    y_batch_eval = data_set['y_batch_eval']


    # Now the trainging begins. To get more information regarding the training part, and the whole program, see
    # "Deep learing for experts" on tensorflows webpage.
    print('Start training')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start = t.time()

    loss_list_train = []
    accuracy_list_eval = []
    accuracy_list_train = []
    iterations = []

    # Number in "range"=number of training iterations
    for i in range(20000):
        # here 100 reandomly selected rows from the training set are extracted
        x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)
        if i % 100 == 0:
            iterations.append(i)
            # from the 100 rows from the training set, the loss function is calculated
            loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
            # To calculate the accuracy, a bigger set of 300 rows are selected
            x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(300, x_batch_eval, y_batch_eval)
            accuracy_value=sess.run(accuracy, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})
            accuracy_value_train=sess.run(accuracy, feed_dict={x: x_batch_sub, y_: y_batch_sub})
            if i % 1000 == 0:
                print('Iteration nr. ', i, 'Acc: ', accuracy_value)
            accuracy_list_eval.append(accuracy_value)
            accuracy_list_train.append(accuracy_value_train)
        sess.run(train_step, feed_dict={x: x_batch_sub, y_: y_batch_sub})

    end=t.time()

    Traningtime = end - start
    print("Trainingtime: " + str(int(Traningtime))+" seconds")

    # Basic plotting of accuracy and training loss function using matplotlib.pyplot. Havn't figured out how to change the fontsize though.
    fig, ax = plt.subplots(2, figsize=(20, 10)) #fig=entire figure, ax=subplots
    ax[0].plot(iterations[0:-1], loss_list_train[0:-1])
    ax[0].set(ylabel='Loss function', xlabel='Iteration')
    ax[1].plot(iterations[0:-1], accuracy_list_train[0:-1])
    ax[1].plot(iterations[0:-1], accuracy_list_eval[0:-1])
    ax[1].set(ylabel='Accuracy', xlabel='Iterations')

    plt.show(fig)

if __name__ == '__main__':
    #main("XBe_01_10_up_to_7.txt", "gun_01_10_up_to_7.txt", 7)
    #main("XBe_5_up_to_7.txt", "gun_5_up_to_7.txt", 7)
    #main("XBe_5_up_to_7_90_percent.txt", "gun_5_up_to_7_90_percent.txt", 7)
    #main("XBe_up_to_7_5MeV_2252068events_digitized.txt", "XB_gun_up_to_7_5MeV_2252068events_digitized.txt", 7)
    #main("XBe_1gun_0dot1_to_10MeV_10000000events_digitized.txt", "XB_gun_1gun_0dot1_to_10MeV_10000000events_digitized.txt", 7)
    #main("XBe_2gun.txt", "gun_2gun.txt", 2)
    #main("XBe_5_up5_liten.txt", "gun_5_up5_liten.txt", 5)
    #main("XBe_up_to_7guns_0dot1_to_10MeV_2200000events_digitized.txt", "XB_gun_up_to_7guns_0dot1_to_10MeV_2200000events_digitized.txt", 7)
    #main("XBe_up_to_7guns_0dot1_to_10MeV_2292267events_sigma_eq_0dot05_not_sup.txt", "XB_gun_up_to_7guns_0dot1_to_10MeV_2292267events_sigma_eq_0dot05_not_sup.txt", 7)
    #main("XBe_up_to_7guns_0dot1_to_10MeV_ca2000000events_digitized_90percent.txt","XB_gun_up_to_7guns_0dot1_to_10MeV_ca2000000events_digitized_90percent.txt", 7,3,1024)
    #main("XBe_up_to_7guns_5MeV_ca2000000events_digitized_90percent.txt","XB_gun_up_to_7guns_5MeV_ca2000000events_digitized_90percent.txt", 7)
    #main("XBe_up_to_7guns_5dot5_to_6dot_6_ca2200000events_digitized_0percent.txt","XB_gun_up_to_7guns_5dot5_to_6dot_6_ca2200000events_digitized_0percent.txt", 7)
    #main("XBe_up_to_7guns_5dot5_to_6dot_6_ca2000000events_digitized_90percent.txt","XB_gun_up_to_7guns_5dot5_to_6dot_6_ca2000000events_digitized_90percent.txt", 7,3,128)
    main("ord_data_set_XB_up_to_7_5MeV_ca2000000events_digitized_90percent.npz", 7,3,128)






