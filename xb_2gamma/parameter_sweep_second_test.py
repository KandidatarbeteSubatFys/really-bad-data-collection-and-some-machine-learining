import tensorflow as tf
import numpy as np
import time as t
import matplotlib.pyplot as plt
import math as m


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


def moving_average(list, points):
    if len(list) < points:
        return m.fsum(list)/len(list)
    else:
        return m.fsum(list[-1-points:-1])/len(list[-1-points:-1])


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
    W_start = tf.Variable(tf.truncated_normal([start_nodes, hidden_nodes], stddev=0.1), dtype=dtyp)
    b_start = tf.Variable(tf.zeros([hidden_nodes]))
    W_end = tf.Variable(tf.truncated_normal([hidden_nodes, end_nodes], stddev=0.1), dtype=dtyp)
    b_end = tf.Variable(tf.zeros([end_nodes]))
    if relu:
        temp = tf.nn.relu(tf.matmul(input, W_start) + b_start)
        temp2 = hidden_laysers(temp, hidden_nodes, nr_hidden_laysers, relu=relu, dtype=dtyp)
        return tf.matmul(temp2, W_end) + b_end
    else:
        temp = tf.matmul(input, W_start) + b_start
        temp2 = hidden_laysers(temp, hidden_nodes, nr_hidden_laysers, relu=relu, dtype=dtyp)
        return tf.matmul(temp2, W_end) + b_end


def main(file_name_x, file_name_y, dep_file_name, nr_nodes_hidden, nr_hidden_laysers):
    print('Initializing variables')
    x = tf.placeholder(dtype=tf.float32, shape=[None, 162])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    y = def_fc_layers(x, 162, 2, nr_nodes_hidden, nr_hidden_laysers)

    loss = tf.reduce_mean(tf.minimum(tf.reduce_sum(tf.square(y-y_), 1), tf.reduce_sum(tf.square(tf.reverse(y, [-1])-y_), 1)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    err = percent_error(y, y_)
    sum = tf.reduce_sum(y, 1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_list = []
    loss_list_train = []
    move_av = [0]

    print('Reading data')
    x_batch = read_data(file_name_x)
    x_batch_eval = x_batch[0:int(0.2*len(x_batch))]
    x_batch_train = x_batch[int(0.2*len(x_batch)):-1]
    y_batch = read_data(file_name_y)
    y_batch_eval = y_batch[0:int(0.2 * len(x_batch))]
    y_batch_train = y_batch[int(0.2 * len(x_batch)):-1]

    print('Start training')
    start = t.time()
    i = 0
    diff = 10
    while m.fabs(diff) > 0.1 and i < 100000:
        x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)
        if i % 100 == 0:
            loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
            x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(100, x_batch_eval, y_batch_eval)
            loss_value = sess.run(loss, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})
            if i % 10000 == 0:
                print('Iteration nr. ', i, 'Loss: ', loss_value)
            loss_list.append(loss_value)
            move_av.append(moving_average(loss_list, 20))
            diff = move_av[-2] - move_av[-1]
        sess.run(train_step, feed_dict={x: x_batch_sub, y_: y_batch_sub})
        i = i + 1
    end = t.time()

    Traningtime = end - start
    loss_end = sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval})
    Training_iterations = i

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
    sum_out = sess.run(sum, feed_dict={x: x_batch_eval})

    out = out1 + out2
    out_corr = out_corr1 + out_corr2
    err_end = sess.run(err, feed_dict={x: x_batch_eval, y_: y_batch_eval})
    mean = np.mean(err_end)
    std = np.std(err_end)

    return loss_end, mean, std, err_end, out, out_corr, loss_list, loss_list_train, Traningtime, Training_iterations, dep, sum_out




def parameter_sweep():
    hidden_nodes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    hidden_layers = [1, 2, 3, 4, 5, 6]
    n = 0

    with open('stats.txt', 'w') as f1:
        with open('loss.txt', 'w') as f2:
            with open('loss_train.txt', 'w') as f3:
                with open('err.txt', 'w') as f4:
                    for laysers in hidden_layers:
                        for nodes in hidden_nodes:
                            loss, mean, std, err, out, out_corr, loss_list, loss_training_list, trainingtime, iterations, dep, sum_out = \
                                main('xb_data_2gamma_isotropic_0.1-10_100000.txt', 'gunTVals_0.1-10.txt', 'sum_of_dep_energies.txt', nodes, laysers)
                            print('Ploting')
                            fig, ax = plt.subplots(2, 2)
                            ax[0, 0].plot(loss_list)
                            ax[0, 0].plot(loss_training_list)
                            ax[0, 0].setlabel(xlabel='Loss function (MeV)^2', ylabel='Iteration x100')
                            ax[0, 1].hist(err, bins='auto')
                            ax[0, 1].setlabel(xlabel='Counts', ylabel='Relative error')
                            ax[1, 0].scatter(out_corr, out, s=0.1)
                            ax[1, 0].plot([0, 10], [0, 10])
                            ax[1, 0].setlabel(xlabel='Gun energy (MeV)', ylabel='Predicted energy (MeV)')
                            ax[1, 1].plot([0, 10], [0, 10])
                            ax[1, 1].setlabel(xlabel='Deposited energy (MeV)', ylabel='Predicted tot energy (MeV)')
                            n = n +1
                            name = str(n) + '.' + str(laysers) + '.' + str(nodes) + '.png'
                            plt.savefig(name)
                            plt.clf()
                            f1.write(str(n) + ' ' + str(laysers) + ' ' + str(nodes) + ' ' + str(loss) + ' ' + str(mean) + ' ' + str(std) + ' \n')
                            loss_str = ''
                            loss_train_str = ''
                            for i in range(len(loss)):
                                loss_str = loss_str + ' ' + str(loss_list[i])
                                loss_train_str = loss_train_str + ' ' + str(loss_training_list[i])
                            f2.write(loss_str + ' \n')
                            f3.write(loss_train_str + ' \n')
                            err_str = ''
                            for i in err:
                                err_str = err_str + ' ' + str(i)
                            f4.write(err_str + ' \n')



if __name__ == '__main__':
    parameter_sweep()