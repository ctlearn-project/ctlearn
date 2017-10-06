import argparse
#import tftables
from mvcnn import mvcnn_fn_2
import tensorflow as tf
from tables import *
import re
import numpy as np
import random
import os

tf.logging.set_verbosity(tf.logging.DEBUG)

TRAIN_BATCH_SIZE = 64

IMAGE_WIDTH = 240
IMAGE_HEIGHT = 240
IMAGE_CHANNELS = 3

STEPS_PER_SUMMARY = 50

STEPS_PER_VALIDATION = 117

STEPS_PER_IMAGE_VIZ = 1000

NUM_IMAGES_EMBEDDING = 700

STEPS_PER_VIZ_EMBED = 1000

STEPS_TO_VIZ_EMBED = 12000

IMAGE_VIZ_MAX_OUTPUTS = 100

def train(data_file,epochs):
    #open file to determine the telescopes included
    f = open_file(data_file, mode = "r", title = "Input file")
    table = f.root.E0.Events_Training
    table_val = f.root.E0.Events_Validation
    
    label_column_name = args.label_col_name
    
    columns_list = table.colnames 
    tels_list = []
    for i in columns_list:
        if re.match("T[0-9]+",i):
            tels_list.append(i)
    num_tel = len(tels_list)
    num_events_train = table.shape[0]
    num_events_val = table_val.shape[0]
    
    training_batch_indices = [i for i in range(num_events_train) if i%TRAIN_BATCH_SIZE == 0 and i+TRAIN_BATCH_SIZE < num_events_train]
    batches_per_epoch = len(training_batch_indices)

    print("Training settings\n*************************************")
    print("Training batch size: ",TRAIN_BATCH_SIZE)
    print("Training batches/steps per epoch: ",batches_per_epoch)
    print("Total # of training steps: ",batches_per_epoch*epochs)
    print("Total number of training events: ",num_events_train)
    print("Total number of validation events: ",num_events_val)
    print("*************************************")
    print("Validation every {} steps".format(STEPS_PER_VALIDATION))
    print("Summary every {} steps".format(STEPS_PER_SUMMARY))
    print("Image visualization summary every {} steps".format(STEPS_PER_IMAGE_VIZ))
    print("Embedding visualization summary every {} steps".format(STEPS_PER_VIZ_EMBED))
    print("*************************************")

    with tf.Graph().as_default():

        images_placeholder = tf.placeholder(tf.float32, shape=(None,num_tel,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS))
        labels_placeholder = tf.placeholder(tf.int8, shape=(None))

        loss,accuracy,logits,predictions = mvcnn_fn_2(images_placeholder,labels_placeholder)
 
        tf.summary.scalar('training_loss', loss)
        tf.summary.scalar('training_accuracy',accuracy)
        merged = tf.summary.merge_all()

        val_loss_op = tf.summary.scalar('validation_loss', loss)
        val_acc_op = tf.summary.scalar('validation_accuracy', accuracy)

        #locate input and 1st layer filter tensors for visualization
        inputs = tf.get_default_graph().get_tensor_by_name("Conv_block/input:0")
        kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'Conv_block/conv1/kernel:0')[0]
        activations = tf.get_default_graph().get_tensor_by_name("Conv_block/conv1/BiasAdd:0")

        inputs_charge_summ_op = tf.summary.image('inputs_charge',tf.slice(inputs,begin=[0,0,0,0],size=[TRAIN_BATCH_SIZE,IMAGE_WIDTH,IMAGE_HEIGHT,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
        inputs_timing_summ_op = tf.summary.image('inputs_timing',tf.slice(inputs,begin=[0,0,0,1],size=[TRAIN_BATCH_SIZE,IMAGE_WIDTH,IMAGE_HEIGHT,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
        filter_summ_op = tf.summary.image('filter',tf.slice(tf.transpose(kernel, perm=[3, 0, 1, 2]),begin=[0,0,0,0],size=[96,11,11,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
        activations_summ_op = tf.summary.image('activations',tf.slice(activations,begin=[0,0,0,0],size=[TRAIN_BATCH_SIZE,58,58,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)

        #global step
        global_step = tf.Variable(0, name='global_step', trainable=False)
        increment_global_step_op = tf.assign(global_step, global_step+1)
        
        #train op
        if args.optimizer == 'adadelta':
            train_op = tf.train.AdadeltaOptimizer(learning_rate=args.lr).minimize(loss)
        else:
            train_op = tf.train.GradientDescentOptimizer(args.lr).minimize(loss)

        #for embeddings visualization
        fetch = tf.get_default_graph().get_tensor_by_name('Classifier/fc7/BiasAdd:0')
        embedding_var = tf.Variable(np.empty((0,4096),dtype=np.float32),name='Embedding_of_fc7',validate_shape=False)
        new_embedding_var = tf.concat([embedding_var,fetch],0)
        update_embedding = tf.assign(embedding_var,new_embedding_var,validate_shape=False)
        empty = tf.Variable(np.empty((0,4096),dtype=np.float32),validate_shape=False)
        reset_embedding = tf.assign(embedding_var,empty,validate_shape=False)

        #training parameter stuff as string tensors
        #learning_rate_tensor = tf.constant(str(args.lr),name="learning_rate")
        #optimizer_tensor = tf.constant(str(args.optimizer),name="optimizer")
        #train_batch_size_tensor = tf.constant(str(TRAIN_BATCH_SIZE),name="train_batch_size")
        #training_events_tensor = tf.constant(str(num_events_train),name="num_events_train")
        #validation_events_tensor = tf.constant(str(num_events_val),name="num_events_val")
        #batches_per_epoch_tensor = tf.constant(str(batches_per_epoch),name="num_batches_per_epoch")
        #epochs_tensor = tf.constant(str(epochs),name="num_epochs")
        #num_tels_tensor = tf.constant(str(num_tel),name="num_telescopes")

        #create supervised session (summary op can be omitted)
        sv = tf.train.Supervisor(
                init_op=tf.global_variables_initializer(),
                logdir=args.logdir,
                checkpoint_basename=args.checkpoint_basename,
                global_step=global_step,
                summary_op=None,
                save_model_secs=600 
                )

        #training loop in session
        #with sv.managed_session() as sess:
        with sv.managed_session() as sess:
            for i in range(batches_per_epoch*epochs):
                if sv.should_stop():
                    break
                feed_dict = {}
                data_batches = []
                for tel in tels_list:
                    data_batches.append(table.read(training_batch_indices[i%batches_per_epoch],training_batch_indices[i%batches_per_epoch]+TRAIN_BATCH_SIZE,field=tel))
                feed_dict[images_placeholder] = np.stack(data_batches,axis=1)
                feed_dict[labels_placeholder] = table.read(training_batch_indices[i%batches_per_epoch],training_batch_indices[i%batches_per_epoch]+TRAIN_BATCH_SIZE,field=label_column_name)
                if i % batches_per_epoch == batches_per_epoch-1:
                    random.shuffle(training_batch_indices)
                #summarize run parameters
                #if i == 1:
                    #params_summary_list = sess.run([learning_rate_tensor,optimizer_tensor,train_batch_size_tensor,training_events_tensor,validation_events_tensor,batches_per_epoch_tensor,epochs_tensor,num_tels_tensor],feed_dict=feed_dict)
                    #for summary in params_summary_list:
                        #sv.summary_computed(sess,summary)
                if i % STEPS_PER_SUMMARY == 0:
                    summ = sess.run(merged,feed_dict=feed_dict)
                    sv.summary_computed(sess, summ)
                if i % STEPS_PER_VALIDATION == 0:                   
                    val_accuracies = []
                    val_losses = []
                    for n in range(num_events_val):
                        val_feed_dict = {}
                        val_data_batches = []
                        for tel in tels_list:
                            val_data_batches.append(table_val.read(n,n+1,field=tel))
                        val_feed_dict[images_placeholder] = np.stack(val_data_batches,axis=1)
                        val_feed_dict[labels_placeholder] = table_val.read(n,n+1,field=label_column_name)

                        val_loss,val_acc = sess.run([loss,accuracy],feed_dict=val_feed_dict)
                        val_accuracies.append(val_acc)
                        val_losses.append(val_loss)

                    mean_val_accuracy = np.mean(val_accuracies)
                    mean_val_loss = np.mean(val_losses)

                    val_acc_summ = tf.Summary()
                    val_acc_summ.value.add(tag="validation_accuracy", simple_value=mean_val_accuracy)

                    val_loss_summ = tf.Summary()
                    val_loss_summ.value.add(tag="validation_loss", simple_value=mean_val_loss)

                    sv.summary_computed(sess, val_loss_summ)
                    sv.summary_computed(sess, val_acc_summ)

                if i % STEPS_PER_IMAGE_VIZ == 0: 
                    filter_summ,inputs_summ,activations_summ = sess.run([filter_summ_op,inputs_charge_summ_op,activations_summ_op],feed_dict=feed_dict)
                    sv.summary_computed(sess,filter_summ)
                    sv.summary_computed(sess,inputs_summ)
                    sv.summary_computed(sess,activations_summ)

                    print("Image summary complete")

                if i % STEPS_PER_VIZ_EMBED == 0:

                    #reset embedding variable to empty
                    sess.run(reset_embedding,feed_dict=feed_dict)

                    for j in range(NUM_IMAGES_EMBEDDING):
                        embed_feed_dict = {}
                        embed_data_batches = []
                        for tel in tels_list:
                            embed_data_batches.append(table_val.read(j,j+1,field=tel))
                        embed_feed_dict[images_placeholder] = np.stack(embed_data_batches,axis=1)
                        embed_feed_dict[labels_placeholder] = table_val.read(j,j+1,field=label_column_name) 
                        
                        result1 = sess.run(fetch, feed_dict=embed_feed_dict)
                        embed_feed_dict[fetch] = result1
                        result2 = sess.run(new_embedding_var,feed_dict=embed_feed_dict)
                        embed_feed_dict[new_embedding_var] = result2
                        result3 = sess.run(update_embedding,feed_dict=embed_feed_dict)
                                           
                    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
                    config.model_checkpoint_dir = os.path.abspath(args.logdir)
                    embedding = config.embeddings.add()
                    embedding.tensor_name = embedding_var.name
                    embedding.metadata_path = os.path.abspath(os.path.join(args.logdir, 'metadata.tsv'))
                    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(sv.summary_writer, config) 

                    #create corresponding metadata file
                    metadata_file = open(embedding.metadata_path, 'w')
                    for k in range(NUM_IMAGES_EMBEDDING):
                        metadata_file.write('{}\n'.format(table_val.read(k,k+1,field=label_column_name)[0]))         
                    metadata_file.close()

                    print("Embedding summary complete")

                sess.run([train_op,increment_global_step_op],feed_dict=feed_dict)

if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Trains on an hdf5 file.')
    parser.add_argument('h5_file', help='path to h5 file containing data')
    parser.add_argument('--optimizer',default='adadelta')
    parser.add_argument('--epochs',default=10000)
    parser.add_argument('--logdir',default='runs/mvcnn1')
    parser.add_argument('--lr',default=0.00001)
    parser.add_argument('--label_col_name',default='gamma_hadron_label')
    parser.add_argument('--checkpoint_basename',default='mvcnn.ckpt')
    parser.add_argument('--no_embedding', action='store_true')
    parser.add_argument('--no_val',action='store_true')
    parser.add_argument('--no_image_summary',action='store_true')
    args = parser.parse_args()

    train(args.h5_file,args.epochs)
