
import tensorflow as tf
import numpy as np
import time as t
import matplotlib.pyplot as plt
import math as m
import itertools as it
import help_funks as hf
import tf_funks as tff

def early_stopp_std(x_batch, y_batch, x, y_, y, step, loss, sess, save_path, update_iterations, max_iterations, partition=0.05,
                   retraining=False):
    """A regularization for the network. First arguments are the data for x and y, then the (already declared) trainingstep and lossfunction.
    Choose how many iterations it can pass before the validation error must have been updated, max iterations and the partition of the data.
    Set retraining to true if the network should be retrained and not restored."""

    x_batch_eval = x_batch[0:int(0.01 * len(x_batch))]          # batch for evaluating
    x_batch_train = x_batch[int(0.01 * len(x_batch)):-1]        # batch for training
    y_batch_eval = y_batch[0:int(0.01 * len(x_batch))]          # x - the input data, y - the correct data
    y_batch_train = y_batch[int(0.01 * len(x_batch)):-1]

    # Need to def new subsets of trainingdata if retraining is true
    if retraining == True:
        x_batch_train = x_batch_train[int(partition * len(x_batch_train)):-1]
        x_batch_eval = x_batch_train[0:int(partition * len(x_batch_train))]

        y_batch_train = y_batch_train[int(partition * len(x_batch)):-1]
        y_batch_eval = y_batch_train[0:int(partition * len(x_batch_train))]

    opt_mean = [float('Inf')] # inf from the start
    opt_value = [float('Inf')]  # inf from the beginning
    i = 0  # counter for iterations
    opt_iteration = 0  # the number of optimal iterations
    j = 0  # counter for how many iterations done without update the convergence condition
    loss_list = []
    loss_list_train = []
    mean_list = []
    lam = 1

    saver = tf.train.Saver()

    print('Start training')
    start = t.time()

    while j < update_iterations and i < max_iterations:         # train for maximum max_iterations or until the opt_value has not been updated for update_iterations
        x_batch_sub, y_batch_sub = hf.gen_sub_set(100, x_batch_train, y_batch_train)       # batch of 100 events

        if i % 100 == 0:
            loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
            x_batch_eval_sub, y_batch_eval_sub = hf.gen_sub_set(800, x_batch_eval, y_batch_eval)
            loss_value = sess.run(loss, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})
            loss_list.append(loss_value)

            out_corr_theta = []
            out_theta = []
            out = sess.run(y, feed_dict={x: x_batch_eval_sub})  # Run the network on the subset.
            for k in range(len(out)):
                index_list = tff.min_sqare_loss_combination(out[k], y_batch_eval_sub[k], lam=lam)  # Find the right permutation
                for p in range(int(len(out[0]) / 2)):                                              # between correct and predicted data.
                    out_theta.append(out[k][2 * p + 1])
                    out_corr_theta.append(y_batch_eval[k][2 * index_list[p] + 1])

            err_theta = [(out_theta[k] - out_corr_theta[k]) for i in range(len(out_theta))]  # Error in cos(theta).

            err_theta=np.array(err_theta)
            mean = np.mean(err_theta)
            mean_list.append(mean)
            std_theta = np.std(err_theta)
#            print(std_theta)

            if std_theta <= opt_value and np.absolute(mean) < np.absolute(opt_mean) :
                print('New optimal std ', std_theta, 'Mean: ', mean)
                j = 0
                opt_iteration = i
                opt_value = std_theta
                opt_mean = mean
                opt_time = t.time()
                if retraining == False:
                    saver.save(sess, save_path)   # save_path ska vara i format "./tmp/model.ckpt"
                    # print("model saved in path: %s" % save_path)
            else:
                j = j + 1
#                print('Iteration', i)

            if i % 1000 == 0:
                print('Iteration nr. ', i, 'Loss: ', loss_value, ' Std: ', std_theta, 'Mean: ', mean)
        sess.run(step, feed_dict={x: x_batch_sub, y_: y_batch_sub})
        i = i + 1

    end = t.time()

    if retraining == False:
        print("Training finished at iteration " + str(i))
        print('Trainingtime until the lowest std', )

        total_iterations = i
        trainingtime = end - start
        trainingtime_opt = opt_time-start
        loss_end = sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval})

        return trainingtime, trainingtime_opt, total_iterations, opt_iteration, loss_end, loss_list, loss_list_train, mean_list, None, None, None, None

    else:    #retrain the network with all trainingdata opt_iteration time

        sess.run(tf.global_variables_initializer())  # reinitialize the parameters

        x_batch_eval = x_batch[0:int(partition * len(x_batch))]
        x_batch_train = x_batch[int(partition * len(x_batch)):-1]
        y_batch_eval = y_batch[0:int(partition * len(x_batch))]
        y_batch_train = y_batch[int(partition * len(x_batch)):-1]

        loss_list2 = []
        loss_list_train2 = []

        print('Start second training to iteration ' + str(opt_iteration))
        start = t.time()
        i = 0
        while i < opt_iteration:
            x_batch_sub, y_batch_sub = hf.gen_sub_set(100, x_batch_train, y_batch_train)
            if i % 100 == 0:
                loss_list_train2.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
                x_batch_eval_sub, y_batch_eval_sub = hf.gen_sub_set(300, x_batch_eval, y_batch_eval)
                loss_value = sess.run(loss, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})
                if i % 10000 == 0:
                    print('Iteration nr. ', i, 'Loss: ', loss_value)
                loss_list2.append(loss_value)
            sess.run(step, feed_dict={x: x_batch_sub, y_: y_batch_sub})
            i = i + 1
        end = t.time()

        trainingtime2 = end - start
        loss_end2 = sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval})

        return trainingtime, trainingtime_opt, total_iterations, opt_iteration, loss_end, loss_list, loss_list_train, mean_list, trainingtime2, loss_end2, loss_list2, loss_list_train2

if __name__ == '__main__':
    print('Det funkar!!')
    print('-- Spring 2018; ce2018')
