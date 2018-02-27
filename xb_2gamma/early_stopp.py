

from parameter_sweep import gen_sub_set
import tensorflow as tf
import time as t

def early_stopping(x_batch, y_batch, x_val, y_val, step, loss, iterations, partition=0.1  , save_parameters=False):
    """A regularization for the network. First arguments are the trainingdata for x and y, then the (already declared) trainingstep and lossfunction.
    Choose how many iterations it can pass before the validation error must have been updated and also the partition of the data.
    Change save_parameter to true if you want to save the optimal parameters."""

    x_batch_eval = x_batch[0:int(partition * len(x_batch))]
    x_batch_train = x_batch[int( partition* len(x_batch)):-1]
    y_batch_eval = y_batch[0:int(partition* len(x_batch))]
    y_batch_train = y_batch[int(partition * len(x_batch)):-1]

    saver = tf.train.Saver()
    sess = tf.Session()
    v = float('Inf')  # vill ha den till inf
    i = 0 # vilken iteration vi är på
    opt_iteration=0 # den bästa iterationen
    j = 0 # så vi vet hur många gånger det har gått innan vi har förbättrat vår valfunction.
    loss_list1 = []


    print('Start first training')
    start = t.time()

    while j < iterations:
        x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)

        if i % 100 == 0:
            #loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
            x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(300, x_batch_eval, y_batch_eval)
            loss_value = sess.run(loss, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})

            if loss_value < v:
                j=0
                opt_iteration=i
                v=loss_value
                if save_parameters:
                    save_path = saver.save(sess, "./tmp/model.ckpt")
                    #print("model saved in path: %s" % save_path)
            else:
                j = j+1

        if i % 10000 == 0:
            print('Iteration nr. ', i, 'Loss: ', loss_value)
            loss_list1.append(loss_value)
        sess.run(step, feed_dict={x: x_batch_sub, y_: y_batch_sub})
        i = i + 1
    end = t.time()

    trainingtime1 = end-start

    # nu vill vi träna om det här nätverket men använda x_batch som träning och x_val som validering

    sess.run(tf.initialize_local_variables) # förhoppningsvis så initialiseras det om nu

    x_batch_eval = x_val
    x_batch_train = x_batch
    y_batch_eval = y_val
    y_batch_train = y_batch

    loss_list2 = []
    loss_list_train = []

    print('Start second training to iteration ' + opt_iteration)
    start = t.time()
    i = 0
    while i < opt_iteration:
        x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)
        if i % 100 == 0:
            loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub}))
            x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(300, x_batch_eval, y_batch_eval)
            loss_value = sess.run(loss, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub})
            if i % 10000 == 0:
                print('Iteration nr. ', i, 'Loss: ', loss_value)
            loss_list2.append(loss_value)
        sess.run(step, feed_dict={x: x_batch_sub, y_: y_batch_sub})
        i = i + 1
    end = t.time()

    trainingtime2 = end - start
    loss_end = sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval})

    return opt_iteration, v, trainingtime1, trainingtime2, loss_list1, loss_list2, loss_list_train, loss_end

