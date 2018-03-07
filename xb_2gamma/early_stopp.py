
from parameter_sweep_second_test import gen_sub_set
import tensorflow as tf
import time as t


def early_stopping(x_batch, y_batch, x, y_, step, loss, sess, iterations, nr_max_iterations, partition=0.05, retraining=False):
	"""A regularization for the network. First arguments are the data for x and y, then the (already declared) trainingstep and lossfunction.
	Choose how many iterations it can pass before the validation error must have been updated, max iterations and the partition of the data.
	Set retraining to true if the network should be retrained and not restored."""

	x_batch_eval = x_batch[0:int(0.01* len(x_batch))]
	x_batch_train = x_batch[int( 0.01* len(x_batch)):-1]
	y_batch_eval = y_batch[0:int(0.01* len(x_batch))]
	y_batch_train = y_batch[int(0.01 * len(x_batch)):-1]
    
    
    # Need to def new subsets of trainingdata if retraining is true
	if retraining == True:
		
		x_batch_train = x_batch_train[int(partition* len(x_batch_train)):-1]
		x_batch_eval = x_batch_train[0:int(partition * len(x_batch_train))]
		
		y_batch_train = y_batch_train[int(partition * len(x_batch)):-1]
		y_batch_eval = y_batch_train[0:int(partition* len(x_batch_train))]
		
	saver = tf.train.Saver()
	opt_value = float('Inf')  # vill ha den till inf
	i = 0 # vilken iteration vi är på
	opt_iteration=0 # den bästa iterationen
	j = 0 # så vi vet hur många gånger det har gått innan vi har förbättrat vår valfunction.
	loss_list = []
	loss_list_train = []


	print('Start training')
	start = t.time()

	while j < iterations and i < nr_max_iterations:
		x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)

		if i % 100 == 0:
			loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
			x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(300, x_batch_eval, y_batch_eval)
			loss_value = sess.run(loss, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})
			loss_list.append(loss_value)
			if loss_value < opt_value:
				j=0
				opt_iteration=i
				opt_value=loss_value
				if retraining == False:
					save_path = saver.save(sess, "./tmp/model.ckpt")
					#print("model saved in path: %s" % save_path)
			else:
				j = j+1

		if i % 10000 == 0:
			print('Iteration nr. ', i, 'Loss: ', loss_value)
		sess.run(step, feed_dict={x: x_batch_sub, y_: y_batch_sub})
		i = i + 1
	end = t.time()
	
	print("Training finished at iteration " + str(i) )
	total_iterations = i
	trainingtime1 = end-start
	loss_end = sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval})
    
	if retraining == False:
		saver.restore(sess, "./tmp/model.ckpt")
		print("Model restored")    
		return opt_iteration, total_iterations, opt_value, trainingtime1, loss_list, loss_list_train, loss_end, None, None, None, None
		
	else:

    # nu vill vi träna om det här nätverket men använda all träningsdata

		sess.run(tf.global_variables_initializer()) # förhoppningsvis så initialiseras det om nu

		x_batch_eval = x_batch[0:int(partition * len(x_batch))]
		x_batch_train = x_batch[int(partition* len(x_batch)):-1]
		y_batch_eval = y_batch[0:int(partition* len(x_batch))]
		y_batch_train = y_batch[int(partition * len(x_batch)):-1]

		loss_list2 = []
		loss_list_train2 = []

		print('Start second training to iteration ' + str(opt_iteration))
		start = t.time()
		i = 0
		while i < opt_iteration:
			x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)
			if i % 100 == 0:
				loss_list_train2.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
				x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(300, x_batch_eval, y_batch_eval)
				loss_value = sess.run(loss, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})
				if i % 10000 == 0:
					print('Iteration nr. ', i, 'Loss: ', loss_value)
				loss_list2.append(loss_value)
			sess.run(step, feed_dict={x: x_batch_sub, y_: y_batch_sub})
			i = i + 1
		end = t.time() 

		trainingtime2 = end - start
		loss_end2 = sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval})
    
		return opt_iteration, total_iterations, opt_value, trainingtime, loss_list, loss_list_train, loss_end, trainingtime2, loss_list2, loss_list_train2, loss_end2
