import tensorflow as tf
import numpy as np
import time as t
import matplotlib.pyplot as plt


def read_data(file):
    out = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
           temp = line.split(' ')
           out.append([float(s) for s in temp[0:-1]])
    return out


def gen_sub_set(batch_size, batch_x, batch_y):
    if not len(batch_x) == len(batch_y):
        raise ValueError('Lists most be of same length /Pontus')
    index_list = np.random.randint(0, len(batch_x), size=batch_size)
    return [batch_x[index] for index in index_list], [batch_y[index] for index in index_list]


def percent_error(x, y):
    return tf.sqrt(tf.divide(tf.minimum(tf.reduce_sum(tf.square(x-y), 1), tf.reduce_sum(tf.square(tf.reverse(x, [-1])-y), 1)), tf.reduce_sum(tf.square(y), 1)))


def hidden_laysers(input, dim, nr, relu=True, dtype=tf.float32):
    W = tf.Variable(tf.truncated_normal([dim, dim], stddev=0.1), dtype=dtype)
    b = tf.Variable(tf.zeros([dim]), dtype=dtype)
    if nr == 0 or nr < 0:
        raise ValueError('Number of laysers most be a posetive integer /Pontus')
    elif nr == 1:
        if relu:
            return tf.nn.relu(tf.matmul(input, W) + b)
        else:
            return tf.matmul(input, W) + b
    else:
        if relu:
            return tf.nn.relu(tf.matmul(hidden_laysers(input, dim, nr-1, relu=relu, dtype=dtype), W) + b )
        else:
            return tf.matmul(hidden_laysers(input, dim, nr-1, relu=relu, dtype=dtype), W) + b


def def_fc_layers(input, start_nodes, end_nodes, hidden_nodes, nr_hidden_laysers, relu=True, dtyp=tf.float32):
    W_start = tf.Variable(tf.truncated_normal([start_nodes, hidden_nodes], stddev=0.1), dtype=tf.float32)
    b_start = tf.Variable(tf.zeros([hidden_nodes]))
    W_end = tf.Variable(tf.truncated_normal([hidden_nodes, end_nodes], stddev=0.1), dtype=tf.float32)
    b_end = tf.Variable(tf.zeros([hidden_nodes]))
    if relu:
        temp = tf.nn.relu(tf.matmul(input, W_start) + b_start)
        temp2 = hidden_laysers(temp, hidden_nodes, nr_hidden_laysers, relu=relu, dtype=dtyp)
        return tf.nn.relu(tf.matmul(temp2, W_end) + b_end)
    else:
        temp = tf.matmul(input, W_start) + b_start
        temp2 = hidden_laysers(temp, hidden_nodes, nr_hidden_laysers, relu=relu, dtype=dtyp)
        return tf.matmul(temp2, W_end) + b_end


def main(file_name_x, file_name_y, dep_file_name, nr_nodes_hidden):
    print('Initializing variables')
    x = tf.placeholder(dtype=tf.float32, shape=[None, 162])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    a = nr_nodes_hidden
    W1 = tf.Variable(tf.truncated_normal([162, a], stddev=0.1), dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([a]))
    W2 = tf.Variable(tf.truncated_normal([a, a], stddev=0.1), dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([a]), dtype=tf.float32)
    W3 = tf.Variable(tf.truncated_normal([a, a], stddev=0.1), dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([a]), dtype=tf.float32)
    W4 = tf.Variable(tf.truncated_normal([a, a], stddev=0.1), dtype=tf.float32)
    b4 = tf.Variable(tf.zeros([a]), dtype=tf.float32)
    W5 = tf.Variable(tf.truncated_normal([a, a], stddev=0.1), dtype=tf.float32)
    b5 = tf.Variable(tf.zeros([a]), dtype=tf.float32)
    W6 = tf.Variable(tf.truncated_normal([a, 2], stddev=0.1), dtype=tf.float32)
    b6 = tf.Variable(tf.zeros([2]), dtype=tf.float32)

    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
    y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
    y4 = tf.nn.relu(tf.matmul(y3, W4) + b4)
    y5 = tf.nn.relu(tf.matmul(y4, W5) + b5)
    y = tf.matmul(y5, W6) + b6

    loss = tf.reduce_mean(tf.minimum(tf.reduce_sum(tf.square(y-y_), 1), tf.reduce_sum(tf.square(tf.reverse(y, [-1])-y_), 1)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    err = percent_error(y, y_)
    sum = tf.reduce_sum(y, 1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_list = []
    loss_list_train = []

    print('Reading data')
    x_batch = read_data(file_name_x)
    x_batch_eval = x_batch[0:int(0.2*len(x_batch))]
    x_batch_train = x_batch[int(0.2*len(x_batch)):-1]
    y_batch = read_data(file_name_y)
    y_batch_eval = y_batch[0:int(0.2 * len(x_batch))]
    y_batch_train = y_batch[int(0.2 * len(x_batch)):-1]

    print('Start training')
    start = t.time()
    for i in range(100000):
        x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)
        if i % 1000 == 0:
            loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
            x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(100, x_batch_eval, y_batch_eval)
            loss_value = sess.run(loss, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})
            print('Iteration nr. ', i, 'Loss: ', loss_value)
            loss_list.append(loss_value)
        sess.run(train_step, feed_dict={x: x_batch_sub, y_: y_batch_sub})
    end = t.time()

    print('Traningtime: ', end - start)
    print('Loss: ', sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval}))
    
    print('Calculating output')
    out1 = []
    out2 = []
    for i in sess.run(y, feed_dict={x: x_batch_eval}):
        out1.append(i[0])
        out2.append(i[1])
    out_corr1 = []
    out_corr2 = []
    for i in range(len(y_batch_eval)):
        if np.power(out1[i]-y_batch_eval[i][0], 2) + np.power(out2[i]-y_batch_eval[i][1], 2) < np.power(out2[i]-y_batch_eval[i][0], 2) + np.power(out1[i]-y_batch_eval[i][1], 2):
            out_corr1.append(y_batch_eval[i][0])
            out_corr2.append(y_batch_eval[i][1])
        else:
            out_corr1.append(y_batch_eval[i][1])
            out_corr2.append(y_batch_eval[i][0])
    dep = [i[0] for i in read_data(dep_file_name)[0:int(0.2*len(x_batch))]]

    print('Ploting')
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(loss_list)
    ax[0, 0].plot(loss_list_train)
    ax[0, 1].hist(sess.run(err, feed_dict={x: x_batch_eval, y_: y_batch_eval}), bins='auto')
    ax[1, 0].scatter(out_corr1, out1, s=0.1)
    ax[1, 0].plot([0, 10], [0, 10])
    ax[1, 1].scatter(out_corr2, out2, s=0.1)
    ax[1, 1].plot([0, 10], [0, 10])
    #ax[1, 1].scatter(dep, sess.run(sum, feed_dict={x: x_batch_eval}), s=0.1)

    plt.show()


if __name__ == '__main__':
    main('xb_data_2gamma_isotropic_0.1-10_100000.txt', 'gunTVals_0.1-10.txt', 'sum_of_dep_energies.txt', 4096)
