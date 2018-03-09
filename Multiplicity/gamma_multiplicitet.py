import tensorflow as tf
import time as t
import numpy as np
import matplotlib.pyplot as plt


#function for reading file of floats with spaces between them. Not the same as Pontus' function, mine might be a little
#slower but I made a new because I thought that Pontus' function was causing problems for me, but it was probably
#something else. Anyway, it works and I havn't bothered switching.
def read_data(file):
    out=[]
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            tmp_string_list = line.split(' ')
            out.append(list(map(float, tmp_string_list)))

    return out


#This function randomly selects some rows from the matrixes batch_x and batch_y and returns the rows as two matrixes.
#The number of rows is determined by the batch_size variable. This is exacly the same function that Pontus wrote.
def gen_sub_set(batch_size, batch_x, batch_y):
    if not len(batch_x) == len(batch_y):
        raise ValueError('Lists most be of same length /Pontus')
    index_list = np.random.randint(0, len(batch_x), size=batch_size)
    return [batch_x[index] for index in index_list], [batch_y[index] for index in index_list]


#This functions is used to get just the energy values and not the cos(theta) values from the gun matrix.
# So it basicly removes every odd column of the gun matrix. This has no real use for the multiplicity data, but
#I used it anyway because maybe I need it later on.
def just_energies_in_gun(y_batch,number_particles):
    tmp=y_batch[:,0]
    for i in range(number_particles-1):
        tmp=np.c_[tmp,y_batch[:,2*i+2]]

    return tmp


def main(file_name_x, file_name_y,number_particles):
    print('Initializing variables')

    #Making placeholders for the inputdata (x) and the correct output data (y_)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 162]) #162=number of crystals in crystal ball detector
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, number_particles+1])

    #Initialization of the weight matrixes and the biases
    W1 = tf.Variable(tf.truncated_normal([162, 128], stddev=0.1), dtype=tf.float32)
    b1 = tf.Variable(tf.ones([128]))
    W2 = tf.Variable(tf.truncated_normal([128, 128], stddev=0.1), dtype=tf.float32)
    b2 = tf.Variable(tf.ones([128]), dtype=tf.float32)
    W3 = tf.Variable(tf.truncated_normal([128, 128], stddev=0.1), dtype=tf.float32)
    b3 = tf.Variable(tf.ones([128]), dtype=tf.float32)
    W4 = tf.Variable(tf.truncated_normal([128, number_particles + 1], stddev=0.1), dtype=tf.float32)
    b4 = tf.Variable(tf.ones([number_particles + 1]), dtype=tf.float32)


    #construction of the layers, using relu activation on the hidden layers
    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
    y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
    y=tf.matmul(y3, W4) + b4

    #As the loss funtion the softmax-crossentropy is used since it's common for classification problems.
    #To optimize the variables, Adam Optimizer is used since it fairs well in comparisons and is easy to use.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    #To check the accuracy, the highest argument of the outputlayer and the one-hot-vector (one-hot is just a way to
    #represent the correct number of guns) is compared

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # converts boelean to ones and zeros and takes the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    #Now begins the creating of data from the textfiles. First the data from
    #the textfiles that came from the simulation is converted into numpy arrays. Then the cos(theta) values from the
    #gun matrix is removed (I could have just extracted the number of guns imidiately, but I wanted to keep the code
    #slightly more general). Then, since we don't want to evaluate on the highest number of guns that was used (for
    # example if you have data for up to 7 guns, then you will train on all the data, but only evaluate on 6) all
    # the other data is extracted. Then this extracted data is divided into a training batch and an evaluation batch.
    #
    #The training data from the extracted data is then merged with the highest-gun data. Data for cases where nothing
    #happends is also added to the training data (gun energy=0, deposited energy=0). Then the data is shuffled (but the
    #corresponing pair of input data and output data is always on the same "row"). Lastly the output data on turned
    #into one-hot vectors because its a good representation of the number of guns used for the loss function.

    print('Reading data')
    x_batch = read_data(file_name_x)
    # quick way to check for number of guns less than there is in the data file by chosing an index to stop on
    #x_batch=x_batch[0:457576]
    x_batch = np.array(x_batch)

    y_batch = read_data(file_name_y)
    #y_batch = y_batch[0:457576]  # remove later
    y_batch = np.array(y_batch)

    y_batch=just_energies_in_gun(y_batch,number_particles)

    #Here the index for when the highest-gun data starts is extracted (the data file must be ordered from lowest amount
    #of guns to highest for this method to work). If the number of particles value is higher than the largest amount of
    #guns that is used, the highest-gun data will also be evaluated on.
    index_last_eval=len(x_batch)-1
    for i in range(y_batch.shape[0]):
        tmp=np.trim_zeros(y_batch[i])
        if tmp.shape[0] == number_particles:
            index_last_eval = i-1
            break

    #All the data except for the data for the highest amount of guns is extracted
    x_batch_1_eval=x_batch[[i for i in range(index_last_eval+1)],:]
    y_batch_1_eval=y_batch[[i for i in range(index_last_eval+1)],:]

    #The extracted data is then shuffled
    shuffle_1_eval=np.c_[x_batch_1_eval,y_batch_1_eval]
    np.random.shuffle(shuffle_1_eval)
    x_batch_1_eval=shuffle_1_eval[:,[i for i in range(162)]]
    y_batch_1_eval = shuffle_1_eval[:, [i for i in range(162,162+number_particles)]]

    #The shuffled extracted data is divided in a training and a evaluation set
    x_batch_eval=x_batch_1_eval[[i for i in range(int(0.2*len(x_batch_1_eval)))],:]
    y_batch_eval = y_batch_1_eval[[i for i in range(int(0.2*len(y_batch_1_eval)))],:]
    x_batch_train_1_eval = x_batch_1_eval[[i for i in range(int(0.2*len(x_batch_1_eval)),len(x_batch_1_eval))],:]
    y_batch_train_1_eval = y_batch_1_eval[[i for i in range(int(0.2*len(y_batch_1_eval)),len(y_batch_1_eval))],:]

    #The training set of the extracted data is merged with the highest-number-of-guns data, as well as with zero events
    x_zeros=np.zeros((int(len(x_batch)/(2*number_particles)),162)) #lägg till minus 1
    y_zeros = np.zeros((int(len(x_batch)/(2*number_particles)),number_particles))
    x_batch_highest_number_of_guns=x_batch[[i for i in range(index_last_eval+1,len(x_batch))],:]
    y_batch_highest_number_of_guns=y_batch[[i for i in range(index_last_eval + 1, len(y_batch))], :]
    x_batch_highest_number_of_guns_decreased = x_batch_highest_number_of_guns[
                                               [i for i in range(int(0.8 * len(x_batch_highest_number_of_guns)))], :]
    y_batch_highest_number_of_guns_decreased = y_batch_highest_number_of_guns[
                                               [i for i in range(int(0.8 * len(y_batch_highest_number_of_guns)))], :]
    x_batch_train=np.r_[np.r_[x_zeros,x_batch_highest_number_of_guns_decreased],x_batch_train_1_eval]
    y_batch_train = np.r_[np.r_[y_zeros, y_batch_highest_number_of_guns_decreased],y_batch_train_1_eval]

    #The training data is shuflled
    shuffle_train = np.c_[x_batch_train, y_batch_train]
    np.random.shuffle(shuffle_train)
    x_batch_train = shuffle_train[:, [i for i in range(162)]]
    y_batch_train = shuffle_train[:, [i for i in range(162,162+number_particles)]]

    # the above was more general, but now for the multiplicity

    #the evaluation input data is turned from a numpy array to a list
    x_batch_eval=x_batch_eval.tolist()
    #indices is the number of guns for each row
    indices = [len(np.trim_zeros(row)) for row in y_batch_eval]
    #the number of guns is now converted into one-hot vectors. Perhaps a bit inefficient to convert to string and then
    #back to float, but this method works nevertheless.
    y_batch_eval=[]
    for i in indices:
        row_str="0 "*i+str(1)+" "+(number_particles-i)*"0 "
        row_str = row_str.rstrip()
        tmp_string_list = row_str.split(' ')
        y_batch_eval.append(list(map(float, tmp_string_list)))


    #same procedure as above
    x_batch_train = x_batch_train.tolist()
    indices = [len(np.trim_zeros(row)) for row in y_batch_train]
    y_batch_train = []
    for i in indices:
        row_str = "0 " * i + str(1) +" " + (number_particles-i)*"0 "
        row_str = row_str.rstrip()
        tmp_string_list = row_str.split(' ')
        y_batch_train.append(list(map(float, tmp_string_list)))


    #To get more information regarding the training part, and the whole program, see "Deep learing for experts" on ten
    #sorflows webpage.
    print('Start training')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start = t.time()

    loss_list_train = []
    accuracy_list_eval = []
    iterations = []

    #Number in "range"=number of training iterations
    for i in range(100000):
        #here 100 reandomly selected rows from the training set are extracted
        x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)
        if i % 100 == 0:
            iterations.append(i)
            #from the 100 rows from the training set, the loss function is calculated
            loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
            #To calculate the accuracy, a bigger set of 300 rows are selected
            x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(300, x_batch_eval, y_batch_eval)
            accuracy_value=sess.run(accuracy, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})
            if i % 1000 == 0:
                print('Iteration nr. ', i, 'Acc: ', accuracy_value)
            accuracy_list_eval.append(accuracy_value)
        sess.run(train_step, feed_dict={x: x_batch_sub, y_: y_batch_sub})

    end=t.time()

    Traningtime = end - start
    print("Trainingtime: " + str(int(Traningtime))+" seconds")

    #Basic plotting using matplotlib.pyplot
    fig, ax = plt.subplots(2, figsize=(20, 10)) #fig=entire figure, ax=subplots
    ax[0].plot(iterations[0:-1], loss_list_train[0:-1])
    ax[0].set(ylabel='Loss function', xlabel='Iteration')
    ax[1].plot(iterations[0:-1], accuracy_list_eval[0:-1])
    ax[1].set(ylabel='Accuracy', xlabel='Iterations')
    plt.show(fig)

if __name__ == '__main__':
    #main("XBe_01_10_up_to_7.txt", "gun_01_10_up_to_7.txt", 7)
    main("XBe_5_up_to_7.txt", "gun_5_up_to_7.txt", 7)
    #main("XBe_2gun.txt", "gun_2gun.txt", 2)
    #main("XBe_5_up5_liten.txt", "gun_5_up5_liten.txt", 5)

