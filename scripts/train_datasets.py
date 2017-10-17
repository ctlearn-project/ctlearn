import sys
import os
import math
#add parent directory to pythonpath to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
from models.mvcnn import mvcnn_fn_2
from models.custom_multi_input import custom_multi_input_v2
import tensorflow as tf
from tables import *
import re
import numpy as np
import random

tf.logging.set_verbosity(tf.logging.DEBUG)

NUM_THREADS = 1
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64

EPOCHS_PER_IMAGE_VIZ = 2
IMAGE_VIZ_MAX_OUTPUTS = 100
EPOCHS_PER_VIZ_EMBED = 2
NUM_BATCHES_EMBEDDING = 20

def train(model,data_file,epochs):

    def load_train_data(index):
        record = table.read(index,index+1)
        tel_imgs = []
        for tel in tels_list:
            tel_imgs.append(record[tel])
        imgs = np.squeeze(np.stack(tel_imgs,axis=1)).astype(np.float32)
        label = record[label_column_name].astype(np.int8)        
        return [imgs, label]

    def load_val_data(index):
        record = table_val.read(index,index+1)
        tel_imgs = []
        for tel in tels_list:
            tel_imgs.append(record[tel])
        imgs = np.squeeze(np.stack(tel_imgs,axis=1)).astype(np.float32)
        label = record[label_column_name].astype(np.int8)
        return [imgs, label]

    #open HDF5 file for reading
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

    #shape of images
    img_shape = table.read(0,1,field=tels_list[0]).shape
    img_width = img_shape[1]
    img_height = img_shape[2]
    img_depth = img_shape[3]

    #data input
    train_dataset = tf.contrib.data.Dataset.range(num_events_train)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.map((lambda index: tuple(tf.py_func(load_train_data, [index], [tf.float32, tf.int8]))),num_threads=NUM_THREADS,output_buffer_size=100*TRAIN_BATCH_SIZE)
    train_dataset = train_dataset.batch(TRAIN_BATCH_SIZE)

    val_dataset = tf.contrib.data.Dataset.range(num_events_val)
    val_dataset = val_dataset.map((lambda index: tuple(tf.py_func(load_val_data,[index],[tf.float32, tf.int8]))), num_threads=NUM_THREADS,output_buffer_size=100*VAL_BATCH_SIZE)
    val_dataset = val_dataset.batch(VAL_BATCH_SIZE)

    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    next_example, next_label = iterator.get_next()

    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(val_dataset)

    print("Training settings\n*************************************")
    print("Training batch size: ",TRAIN_BATCH_SIZE)
    print("Validation batch size: ",VAL_BATCH_SIZE)
    print("Training batches/steps per epoch: ",math.ceil(num_events_train/TRAIN_BATCH_SIZE))
    print("Total # of training steps: ",math.ceil(num_events_train/TRAIN_BATCH_SIZE)*epochs)
    print("Total number of training events: ",num_events_train)
    print("Total number of validation events: ",num_events_val)
    print("*************************************")
    print("Image visualization summary every {} epochs".format(EPOCHS_PER_IMAGE_VIZ))
    print("Embedding visualization summary every {} epochs".format(EPOCHS_PER_VIZ_EMBED))
    print("*************************************")

    loss,accuracy,logits,predictions = model(next_example,next_label)

    tf.summary.scalar('training_loss', loss)
    tf.summary.scalar('training_accuracy',accuracy)
    merged = tf.summary.merge_all()

    #locate input and 1st layer filter tensors for visualization
    inputs = tf.get_default_graph().get_tensor_by_name("Conv_block_T0/input:0")
    kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'Conv_block_T0/conv1/kernel:0')[0]
    activations = tf.get_default_graph().get_tensor_by_name("Conv_block_T0/conv1/BiasAdd:0")

    inputs_charge_summ_op = tf.summary.image('inputs_charge',tf.slice(inputs,begin=[0,0,0,0],size=[TRAIN_BATCH_SIZE,img_width,img_height,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
    inputs_timing_summ_op = tf.summary.image('inputs_timing',tf.slice(inputs,begin=[0,0,0,1],size=[TRAIN_BATCH_SIZE,img_width,img_height,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
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
        for i in range(epochs):
            sess.run(training_init_op)
            print("Epoch {} started...".format(i+1))
            while True:
                try:
                    sess.run([train_op,increment_global_step_op])
                    summ = sess.run(merged)
                    sv.summary_computed(sess, summ)
                except tf.errors.OutOfRangeError:
                    break

            print("Epoch {} finished. Validating...".format(i+1))

            #validation
            sess.run(validation_init_op)
            val_accuracies = []
            val_losses = []
            while True:
                try:
                    val_loss,val_acc = sess.run([loss,accuracy])
                    val_accuracies.append(val_acc)
                    val_losses.append(val_loss)
                except tf.errors.OutOfRangeError:
                    break

            mean_val_accuracy = np.mean(val_accuracies)
            mean_val_loss = np.mean(val_losses)
            val_acc_summ = tf.Summary()
            val_acc_summ.value.add(tag="validation_accuracy", simple_value=mean_val_accuracy)
            val_loss_summ = tf.Summary()
            val_loss_summ.value.add(tag="validation_loss", simple_value=mean_val_loss)
            sv.summary_computed(sess, val_loss_summ)
            sv.summary_computed(sess, val_acc_summ)

            print("Validation complete.")

            if i % EPOCHS_PER_IMAGE_VIZ == 0: 
                filter_summ,inputs_summ,activations_summ = sess.run([filter_summ_op,inputs_charge_summ_op,activations_summ_op])
                sv.summary_computed(sess,filter_summ)
                sv.summary_computed(sess,inputs_summ)
                sv.summary_computed(sess,activations_summ)

                print("Image summary complete")

            if i % EPOCHS_PER_VIZ_EMBED == 0:

                #reset embedding variable to empty
                sess.run(reset_embedding)
                
                for j in range(NUM_BATCHES_EMBEDDING):
                    try:
                        sess.run(fetch)
                        sess.run(new_embedding_var)
                        sess.run(update_embedding)
                    except tf.errors.OutOfRangeError:
                        break

                                      
                config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
                config.model_checkpoint_dir = os.path.abspath(args.logdir)
                embedding = config.embeddings.add()
                embedding.tensor_name = embedding_var.name
                embedding.metadata_path = os.path.abspath(os.path.join(args.logdir, 'metadata.tsv'))
                tf.contrib.tensorboard.plugins.projector.visualize_embeddings(sv.summary_writer, config) 

                #write corresponding metadata file
                metadata_file = open(embedding.metadata_path, 'w')
                for k in range(NUM_IMAGES_EMBEDDING):
                    metadata_file.write('{}\n'.format(table_val.read(k,k+1,field=label_column_name)[0]))         
                metadata_file.close()

                print("Embedding summary complete")

if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Trains on an hdf5 file.')
    parser.add_argument('h5_file', help='path to h5 file containing data')
    parser.add_argument('--optimizer',default='adadelta')
    parser.add_argument('--epochs',default=10000)
    parser.add_argument('--logdir',default='/data0/logs/custom_multi_input_datasets_test')
    parser.add_argument('--lr',default=0.00001)
    parser.add_argument('--label_col_name',default='gamma_hadron_label')
    parser.add_argument('--checkpoint_basename',default='custom_multi_input.ckpt')
    parser.add_argument('--no_embedding', action='store_true')
    parser.add_argument('--no_val',action='store_true')
    parser.add_argument('--no_image_summary',action='store_true')
    args = parser.parse_args()

    train(custom_multi_input_v2,args.h5_file,args.epochs)
