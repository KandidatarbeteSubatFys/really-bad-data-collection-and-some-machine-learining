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


def percent_tile_network(percent, x, y):
    per2 = tf.square(percent_error(x, y))
    per2_lim  = tf.constant(percent*percent, shape=tf.shape(x), dtype=tf.float32)
    return tf.reduce_mean(tf.cast(tf.less(per2, per2_lim),dtype=tf.float32))


def percent_error(x, y):
    return tf.divide(x-y, y)


def main(file_name_x, file_name_y, dep_file_name):
    print('Initializing variables ')
    x = tf.placeholder(dtype=tf.float32, shape=[None, 162])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    W1 = tf.Variable(tf.truncated_normal([162, 1024], stddev=0.1), dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([1024]))
    W2 = tf.Variable(tf.truncated_normal([162, 1], stddev=0.1), dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32)

    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    y = tf.matmul(x, W2) + b2

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-y_), 1))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    err = percent_error(y, y_)
    #p10 = percent_tile_network(0.1, y, y_)
    #p5 = percent_tile_network(0.05, y, y_)
    #p1 = percent_tile_network(0.01, y, y_)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_list = []
    loss_list_train = []
    p10_list = []
    p5_list = []
    p1_list = []

    print('Reading data')
    x_batch = read_data(file_name_x)
    x_batch_eval = x_batch[0:int(0.2*len(x_batch))]
    x_batch_train = x_batch[int(0.2*len(x_batch)):-1]
    y_batch = read_data(file_name_y)
    y_batch_eval = y_batch[0:int(0.2 * len(x_batch))]
    y_batch_train = y_batch[int(0.2 * len(x_batch)):-1]

    print('Start training')
    start = t.time()
    for i in range(50000):
        x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)
        if i % 100 == 0:
            loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
            x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(100, x_batch_eval, y_batch_eval)
            loss_value = sess.run(loss, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})
            print('Iteration nr. ', i, 'Loss: ', loss_value)
            loss_list.append(loss_value)
            #p10_list.append(sess.run(p10, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub}))
            #p5_list.append(sess.run(p5, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub}))
            #p1_list.append(sess.run(p1, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub}))
        sess.run(train_step, feed_dict={x: x_batch_sub, y_: y_batch_sub})
    end = t.time()

    print('Traningtime: ', end - start)
    print('Loss: ', sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval}))
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(loss_list)
    ax[0, 0].plot(loss_list_train)
    ax[0, 0].set(xlabel='Iterations in hundreds', ylabel='Loss function value (MeV^2)')
    
    #axis[1, 0].plot(p10_list)
    #axis[1, 0].plot(p5_list)
    #axis[1, 0].plot(p1_list)

    ax[0, 1].hist(sess.run(err, feed_dict={x: x_batch_eval, y_: y_batch_eval}), bins='auto')
    ax[0 ,1].set(xlabel='Relative error', ylabel='Counts')
    out = [i[0] for i in sess.run(y, feed_dict={x: x_batch_eval})]
    dep = [i[0] for i in read_data(dep_file_name)[0:int(0.2 * len(x_batch))]]
    ax[1, 0].scatter([i[0] for i in y_batch_eval], out, s=0.1)
    ax[1, 0].set(xlabel='Gun energy (correct) (MeV)', ylabel='Predicted energy (MeV)')
    ax[1, 1].scatter(dep, out, s=0.1)
    ax[1, 1].set(xlabel='Measured deposited energy (MeV)', ylabel='Predicted energy (MeV)')

    fig.suptitle('XB gamma, T=0.1-10MeV, 100 000 events')

    plt.show()


if __name__ == '__main__':
    main("xb_data_gamma_isotropic_0.1-10_100000.txt", "gunTVals_0.1-10.txt", "sum_of_dep_energies.txt")














