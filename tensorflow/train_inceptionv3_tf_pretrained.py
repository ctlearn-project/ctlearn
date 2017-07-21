import tensorflow as tf
import os
import tftables
from tensorflow.contrib.slim.python.slim.nets import inception_v3
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import layers as layers_lib
import math

with tf.Session() as sess:

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):

        inceptionv3 = inception_v3.inception_v3

        reader = tftables.open_file(filename='/home/gemini/code/imageExtractor/ctapipe/HDF5/output_combined.h5', batch_size=32)

        # Use get_batch to access the table.
        # Both datasets must be accessed in ordered mode.
        table_batch_dict = reader.get_batch(
                path = '/0/tel_data/5',
                ordered = True)
        telescope_data = tf.to_float(tf.transpose(table_batch_dict,[0,2,3,1]))

        #print(telescope_data.shape)

        # Now use get_batch again to access an array.
        # Both datasets must be accessed in ordered mode.
        labels_batch = reader.get_batch('/0/gamma_hadron_label', ordered = True)
        labels = tf.to_float(tf.one_hot(labels_batch,2,1,0))

        #print(labels.shape)

        # The loader takes a list of tensors to be stored in the queue.
        # When accessing in ordered mode, threads should be set to 1.
        loader = reader.get_fifoloader(
                queue_size = 10,
                inputs = [telescope_data,labels],
                threads = 1)

        # Batches are taken out of the queue using a dequeue operation.
        # Tensors are returned in the order they were given when creating the loader.
        X,y = loader.dequeue()

        #print(X.shape)
        #print(y.shape)

        # Create the model
        predictions = inceptionv3(X,
                num_classes=2,
                is_training=True,
                dropout_keep_prob=0.8,
                min_depth=16,
                depth_multiplier=1.0,
                prediction_fn=layers_lib.softmax,
                spatial_squeeze=True,
                reuse=None,
                scope='InceptionV3')

        #print(predictions[0])

        tf.losses.softmax_cross_entropy(y,predictions[0])

        total_loss = tf.losses.get_total_loss()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Specify where the Model, trained on ImageNet, was saved.
        model_path = '/home/gemini/code/deep_learning/tensorflow/inception_v3.ckpt'

        # Specify where the new model will live:
        log_dir = '/home/gemini/code/deep_learning/tensorflow'
        # Restore only the convolutional layers:
        variables_to_restore = slim.get_variables_to_restore(exclude=['InceptionV3/AuxLogits/Conv2d_2a_3x3','Predictions','InceptionV3/AuxLogits/Conv2d_2a_3x3/BatchNorm','InceptionV3/Logits/Conv2d_1c_1x1','AuxLogits','InceptionV3/AuxLogits/Conv2d_2b_1x1'])
        #print(variables_to_restore)
        init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # Start training.
        slim.learning.train(train_op, log_dir, init_fn=init_fn)


        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': slim.metrics.accuracy(predictions, labels),
            'precision': slim.metrics.precision(predictions, labels),
            })

        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
                op = tf.Print(op, [metric_value], metric_name)
                  summary_ops.append(op)

                  num_examples = 460
                  batch_size = 20
                  num_batches = math.ceil(num_examples / float(batch_size))

                  # Setup the global step.
                  slim.get_or_create_global_step()

                  output_dir =  # Where the summaries are stored.
                  eval_interval_secs =  # How often to run the evaluation.
                  slim.evaluation.evaluation_loop(
                          'local',
                          checkpoint_dir,
                          log_dir,
                          num_evals=num_batches,
                          eval_op=names_to_updates.values(),
                          summary_op=tf.summary.merge(summary_ops),
                          eval_interval_secs=eval_interval_secs)
