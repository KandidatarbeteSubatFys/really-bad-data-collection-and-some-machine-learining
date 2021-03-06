import tensorflow as tf
import numpy as np
import time as t
import matplotlib.pyplot as plt
import math as m
import itertools as it
import help_funks as hf
import tf_funks as tff
from matplotlib.colors import LogNorm


def def_network(number_of_particles, nr_nodes_hidden, nr_hidden_layers, lam=1, off_set=0., tree=True):
    print('Initializing variables')
    x = tf.placeholder(dtype=tf.float32, shape=[None, 162])  # Placeholder for input.
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2 * number_of_particles])  # Placeholder for correct output.

    y = tff.def_fc_layers(x, 162, 2 * number_of_particles, nr_nodes_hidden,
                          nr_hidden_layers)  # Defines a fully connected graf.

    loss = tff.split_energy_angle_comb(y, y_, number_of_particles, lam=lam, off_set=off_set, tree=tree)  # Defines loss.
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)  # Set the training method to Adam opt.

    return x, y, y_, loss, train_step


def main(x_batch, y_batch, dep_batch, nr_nodes_hidden, nr_hidden_laysers, number_of_particles, save_name_network, lam=1, tree=True, off_set=0., n=1):
    save = True
    restore = True

    x, y, y_, loss, train_step = def_network(number_of_particles=number_of_particles, nr_nodes_hidden=nr_nodes_hidden,
                                             nr_hidden_layers=nr_hidden_laysers, lam=1, off_set=off_set, tree=tree)

    if save or restore:
        saver = tf.train.Saver()

    sess = tf.Session()
    if restore:
        restore_name = save_name_network
        saver.restore(sess, restore_name)
    else:
        sess.run(tf.global_variables_initializer())


    loss_list = []                                                              # A list to store the loss values.
    loss_list_train = []                                                        # List of loss value for training data.
    move_av = [0]

    print('Partitioning the data')
    x_batch_eval = x_batch[0:int(0.01 * len(x_batch))]                          # 1% as evaluation data set.
    x_batch_train = x_batch[int(0.01 * len(x_batch)):-1]                        # Rest as training data set.
    y_batch_eval = y_batch[0:int(0.01 * len(x_batch))]
    y_batch_train = y_batch[int(0.01 * len(x_batch)):-1]
    dep_batch_eval = dep_batch[0:int(0.01 * len(x_batch))]                      # Deposited energy i detector.

    print('Start training')
    start = t.time()                                                            # The time at the start of the training.
    i = 0
    diff = 10                                                                   # Old converges conditions.
    tol = 0.000001
    while m.fabs(diff) > tol and i < 1e5:                                                        # Train 500000 steps
        x_batch_sub, y_batch_sub = hf.gen_sub_set(100, x_batch_train, y_batch_train)                # Batch size = 100
        if i % 100 == 0:                                                                            # Store statistics.
            loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))     # Train batch loss.
            x_batch_eval_sub, y_batch_eval_sub = hf.gen_sub_set(300, x_batch_eval, y_batch_eval)    # Evaluation batch 300
            loss_value = sess.run(loss, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})      # Eval batch loss.
            if i % 10000 == 0:                                                                      # Print to terminal.
                print('Iteration nr. ', i, 'Loss: ', loss_value)
            loss_list.append(loss_value)
            move_av.append(hf.moving_average(loss_list, 20))                                        # Moving average (Old)
            #diff = move_av[-2] - move_av[-1]
        sess.run(train_step, feed_dict={x: x_batch_sub, y_: y_batch_sub})                           # One train step.
        i = i + 1
    end = t.time()                                                                              # Time when training done.

    Traningtime = end - start
    loss_end = sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval})                    # Loss on the whole set.
    Training_iterations = i

    print('Calculating output')
    out_E = []
    out_theta = []
    out_corr_E = []
    out_corr_theta = []
    out = sess.run(y, feed_dict={x: x_batch_eval})                                      # Run the network on the evalset.
    for i in range(len(out)):
        index_list = tff.min_sqare_loss_combination(out[i], y_batch_eval[i], lam=lam, off_set=off_set)  # Find the right
        for j in range(int(len(out[0])/2)):                                             # permutation between correct
            out_E.append(out[i][2*j])                                                   # and predicted data.
            out_theta.append(out[i][2*j+1])
            out_corr_E.append(y_batch_eval[i][2*index_list[j]])
            out_corr_theta.append(y_batch_eval[i][2*index_list[j]+1])

    # Deposited energy i the detector.
    dep = [dep_val[0] for dep_val in dep_batch_eval]                                    # As a normal list.
    sum_pred = [np.sum(event[[i for i in range(0, len(event), 2)]]) for event in out]   # Sum of predicted energy.

    err_theta = [(out_theta[i]-out_corr_theta[i]) for i in range(len(out_theta))]       # Error in cos(theta).
    err_E = [(out_E[i] - out_corr_E[i])/(out_corr_E[i] + off_set) for i in range(len(out_E))] # Relative error in energy.
    mean_E = np.mean(err_E)                                                             # Mean of energy error.
    mean_theta = np.mean(err_theta)                                                     # Mean of cos(theta) error.
    std_E = np.std(err_E)                                                       # Standard divination for energy error.
    std_theta = np.std(err_theta)                                               # Standard diviation of cos(theta) error.

    if save:
        print('Saving model')
        save_name = './model3.ckpt'
        save_path = saver.save(sess, save_name)


    return loss_end, mean_E, mean_theta, std_E, std_theta, err_E, err_theta, out_E, out_corr_E, out_theta,\
           out_corr_theta, loss_list, loss_list_train, Traningtime, Training_iterations,\
           sum_pred, dep


def plot(iterations, loss_training_list, loss_list, err_E, err_theta, out_corr_E, out_E, dep, sum_pred,
            out_corr_theta, out_theta, nodes, layers, n, fig_path, off_set=0):
    print('Ploting')
    # Set up a subfigure environment.
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))

    # Ploting the lossfunction as a function over traning iterations. Start 39 points in for beater formatting.
    iter_list = [i for i in range(0, iterations, 100)]
    ax[0, 0].plot(iter_list[39:-1], loss_training_list[39:-1])
    ax[0, 0].plot(iter_list[39:-1], loss_list[39:-1])
    ax[0, 0].set(ylabel='Loss function (MeV)^2', xlabel='Iteration')

    # Histogram over the relative errors in energy.
    ax[0, 1].hist(err_E, bins='auto')
    ax[0, 1].set(ylabel='Counts', xlabel='Relative error energy')

    # Histogram over error in cos(theta)
    ax[0, 2].hist(err_theta, bins='auto')
    ax[0, 2].set(ylabel='Counts', xlabel='Error cos(theta)')

    # Plot the relative error against cos(theta)
    h = ax[0, 3].hist2d(out_corr_theta, err_E, bins=int(np.sqrt(len(err_E)/2)), norm=LogNorm())
    ax[0, 3].set(xlabel='Cos(theta) (correct)', ylabel='Relative error energy')
    plt.colorbar(h[3], ax=ax[0, 3])

    # 2D histogram, density plot, predicted against correct.
    h = ax[1, 0].hist2d(out_corr_E, out_E, bins=int(np.sqrt(len(out_E)/2)), norm=LogNorm())
    ax[1, 0].plot([0, 10], [0, 10], 'r')
    ax[1, 0].set(xlabel='Gun energy (MeV)', ylabel='Predicted energy (MeV)')
    plt.colorbar(h[3], ax=ax[1, 0])

    # 2D histogram, density plot, sum predicted energy  against deposited.
    h = ax[1, 1].hist2d(dep, sum_pred, bins=int(np.sqrt(len(dep))), norm=LogNorm())
    ax[1, 1].plot([0, 50], [0, 50], 'r')
    ax[1, 1].set(xlabel='Deposited energy (MeV)', ylabel=' Sum predicted energy (MeV)')
    plt.colorbar(h[3], ax=ax[1, 1])

    # 2D histogram, density plot, predicted cos(theta) against cos(theta) correct.
    h = ax[1, 2].hist2d(out_corr_theta, out_theta, bins=int(np.sqrt(len(out_theta)/2)), norm=LogNorm())
    ax[1, 2].plot([-1, 1], [-1, 1], 'r')
    ax[1, 2].set(xlabel='Correct cos(theta)', ylabel='Predicted cos(theta)')
    plt.colorbar(h[3], ax=ax[1, 2])

    # Plot the error in cos(theta) against energy
    h = ax[1, 3].hist2d(out_corr_E, err_theta, bins=int(np.sqrt(len(err_E)/2)), norm=LogNorm())
    ax[1, 3].set(xlabel='Gun energy (correct) (MeV)', ylabel='Error cos(theta)')
    plt.colorbar(h[3], ax=ax[1, 3])

    # Add a title to the figures containing tha architecture of the network used.
    fig.suptitle(
        'Hidden layers: ' + str(layers) + ', nodes per layer: ' + str(nodes) + ' iterations: ' + str(iterations) +
        ' off set:' + str(off_set))
    name = str(n) + '.' + str(layers) + '.' + str(nodes) + '.png'
    plt.savefig(fig_path + name)                                    # Save the plot
    plt.clf()                                                       # Close the plot


def do_train(nodes, layers, number_of_particles, save_name_network):
    # File name of deposited energy in the crystals.
    energy_file_name = '/net/home/oljacob/ggland20180401/land02/scripts/ggland/train_data_addback_compare/XB_det_data_1:3gamma_0.05:10MeV_isotropic(2).txt'
    # File name of the data from the guns.
    gun_file_name = '/net/home/oljacob/ggland20180401/land02/scripts/ggland/train_data_addback_compare/XB_gun_vals_1:3gamma_0.05:10MeV_isotropic(2).txt'
    # File name of all deposited energy in the detector.
    dep_file_name = '/net/home/oljacob/ggland20180401/land02/scripts/ggland/train_data_addback_compare/XB_sum_of_dep_energies_1:3gamma_0.05:10MeV_isotropic(2).txt'
    # Path to the folder where all figures will be placed.
    fig_path = './figs/'
    # Path to the folder where all non figure files will be placed, note that existing files in this location will be
    # over written if over_write = True.
    file_path = '.'

    # Open files and make dedicated functions to write in them.
    stats_write = hf.make_write_function(file_path=file_path + '/stats.txt', over_write=True, header='')
    loss_write = hf.make_write_function(file_path=file_path + '/loss.txt', over_write=True, header='')
    loss_train_write = hf.make_write_function(file_path=file_path + '/loss_train.txt', over_write=True, header='')
    err_write = hf.make_write_function(file_path=file_path + '/err.txt', over_write=True, header='')

    # Read the training, evaluation and deposited data in to memory.
    print('Reading data')
    x_batch = hf.read_data(energy_file_name, cols=162)
    y_batch = hf.read_data(gun_file_name, cols=2*number_of_particles)
    dep_batch = hf.read_data(dep_file_name, cols=1)

    x_batch, y_batch, dep_batch = hf.randomize_content(x_batch, y_batch, dep_batch)

    # Calls the main script to preform the training.
    loss, mean_E, mean_theta, std_E, std_theta, err_E,\
    err_theta, out_E, out_corr_E, out_theta,\
    out_corr_theta, loss_list,\
    loss_training_list, trainingtime, iterations,\
    sum_pred, dep = main(x_batch, y_batch, dep_batch, nodes, layers, number_of_particles, save_name_network, tree=True, off_set=0.1, n=1)

    # Plots the summarizing plots of the training.
    plot(iterations, loss_training_list, loss_list, err_E, err_theta, out_corr_E, out_E, dep, sum_pred,
        out_corr_theta, out_theta, nodes, layers, 1, fig_path, off_set=0.1)

    # Write to stats file .
    stats_write(str(1) + ' ' + str(layers) + ' ' + str(nodes) + ' ' + str(loss) + ' ' + str(mean_E)
                + ' ' + str(mean_theta) + ' ' + str(std_E) + ' ' + str(std_theta)
                + ' ' + str(trainingtime) + ' ' + str(iterations) + ' \n')

    # Restructure the list of loss values to a string, to be writen to file.
    loss_str = str()
    loss_train_str = str()
    for i in range(len(loss_list)):
        loss_str = loss_str + ' ' + str(loss_list[i])
        loss_train_str = loss_train_str + ' ' + str(loss_training_list[i])

    loss_write(loss_str + ' \n')                                            # Write to loss file.
    loss_train_write(loss_train_str + ' \n')                                # Writ to loss_train file.

    print('Exiting training')


def xb_analysis(nodes, layers, number_of_particles, restore_file_name, threshold):
	energy_file_name = '/net/home/oljacob/ggland20180401/land02/scripts/ggland/train_data_addback_compare/XB_det_data_from_xb_cases_5.0MeV.txt'
	data_file_path = '/net/home/oljacob/ggland20180401/land02/scripts/ggland/xb_sim/addback/neuralnetwork'
	eventnos_filename = '/net/home/oljacob/ggland20180401/land02/scripts/ggland/train_data_addback_compare/XB_eventnos_from_xb_cases_5.0MeV.txt'

	print('Reading energy data')
	x_data = hf.read_data(energy_file_name, cols=162)
	print('Reading eventno data')
	eventnos = hf.read_data(eventnos_filename, cols=1)
	
	print('Restoring network')
	x, y, y_, loss, train_step = def_network(number_of_particles, nodes, layers, lam=1, off_set=0., tree=True)

	saver = tf.train.Saver()
	sess = tf.Session()
	saver.restore(sess, restore_file_name)
	
	out_net = []
	out_corr = []
    
    
	print('Makes analaysis and prints to ' + data_file_path + '/xb_line_E5.0MeV.txt')
	with open(data_file_path + '/xb_line_E5.0MeV.txt', 'w') as f:
		for i, batch in enumerate(np.split(x_data, [i for i in range(10000, len(x_data), 10000)])):
			out = sess.run(y, feed_dict={x: batch})
			for k, event in enumerate(eventnos[i*10000:i*10000+len(out)]):
				f.write('Event: ' + str(int(event[0])) + ' \n')
				for j in range(0, len(out[k]), 2):
					if out[k, j] > threshold:
						string = '  gamma: e={:6.2f} theta={:6.2f} main_c={:3} num_c={:3} \n'.format(out[k, j], np.arccos(out[k, j+1])*180/np.pi, 0, 0)
						f.write(string)
				f.write('\n')
				
	print('Exiting')




if __name__ == '__main__':
    save_network = './model3.ckpt'
    nodes = 256
    layers = 10
    number_of_particles = 3
    do_train(nodes, layers, number_of_particles, save_network)
    #xb_analysis(nodes, layers, number_of_particles, save_network, 0.005)

