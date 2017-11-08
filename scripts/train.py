import sys
import os
import math
import argparse
import re
import random

#add parent directory to pythonpath to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

#disable info and warning messages (not error messages)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tables import *
import numpy as np

from models.mvcnn import mvcnn_fn_2
from models.custom_multi_input import custom_multi_input_v2
from models.variable_input_model import variable_input_model

NUM_THREADS = 12
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64

EPOCHS_PER_IMAGE_VIZ = 5
IMAGE_VIZ_MAX_OUTPUTS = 100
EPOCHS_PER_VIZ_EMBED = 5
NUM_BATCHES_EMBEDDING = 20

def train(model,data_file,epochs,image_summary,embedding):

    def load_data(record):
        tel_map = record["tel_map"][0]
        #print(tel_map)
        tel_imgs = []
        assert tel_map.shape[0] == len(tels_list)
        for i in range(len(tels_list)):
            if tel_map[i] != -1:
                array = f.root.E0._f_get_child(tels_list[i])
                tel_imgs.append(array[tel_map[i]])
            else:
                tel_imgs.append(np.empty([img_width,img_length,img_depth]))

        imgs = np.stack(tel_imgs,axis=0).astype(np.float32)
        label = record[label_column_name].astype(np.int8)
        #convert CORSIKA particle type code to gamma-proton label (0 = proton, 1 = gamma)
        if label[0] == 0:
            label[0] = 1
        elif label[0] == 101:
            label[0] = 0
        trig_list = tel_map
        trig_list[trig_list >= 0] = 1
        trig_list[trig_list == -1] = 0
        trig_list = trig_list.astype(np.int8)
        return [imgs,label,trig_list]

    def load_train_data(index):
        record = table_train.read(index, index + 1)
        return load_data(record)
    
    def load_val_data(index):
        record = table_val.read(index, index + 1)
        return load_data(record)

    #open HDF5 file for reading
    f = open_file(data_file, mode = "r", title = "Input file")
    table_train = f.root.E0.Events_Training
    table_val = f.root.E0.Events_Validation    
    label_column_name = args.label_col_name
    num_events_train = table_train.shape[0]
    num_events_val = table_val.shape[0]

    #prepare constant telescope position vector
    table_telpos = f.root.Tel_Table
    tel_pos_vector = []
    tels_list = []
    for row in table_telpos.iterrows():
        tels_list.append("T" + str(row["tel_id"]))
        tel_pos_vector.append(row["tel_x"])
        tel_pos_vector.append(row["tel_y"])
    num_tel = len(tels_list) 
    tel_pos_tensor = tf.reshape(tf.constant(tel_pos_vector),[num_tel,2])

    #shape of images
    img_shape = eval("f.root.E0.{}.shape".format(tels_list[0]))
    img_width = img_shape[1]
    img_length = img_shape[2]
    img_depth = img_shape[3]

    #create datasets
    train_dataset = tf.contrib.data.Dataset.range(num_events_train)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.map((lambda index: tuple(tf.py_func(load_train_data, [index], [tf.float32, tf.int8, tf.int8]))),num_threads=NUM_THREADS,output_buffer_size=100*TRAIN_BATCH_SIZE)
    train_dataset = train_dataset.batch(TRAIN_BATCH_SIZE)

    val_dataset = tf.contrib.data.Dataset.range(num_events_val)
    val_dataset = val_dataset.map((lambda index: tuple(tf.py_func(load_val_data,[index],[tf.float32, tf.int8, tf.int8]))), num_threads=NUM_THREADS,output_buffer_size=100*VAL_BATCH_SIZE)
    val_dataset = val_dataset.batch(VAL_BATCH_SIZE)

    #create iterator and init ops
    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    next_example, next_label, next_trig_list = iterator.get_next()

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
    if image_summary:
        print("Image visualization summary every {} epochs".format(EPOCHS_PER_IMAGE_VIZ))
    else:
        print("No image visualization")
    if embedding:
        print("Embedding visualization summary every {} epochs".format(EPOCHS_PER_VIZ_EMBED))
    else:
        print("No embedding visualization")
    print("*************************************")

    training = tf.placeholder(tf.bool, shape=())

    loss, accuracy, logits, predictions = model(next_example, next_label, 
            next_trig_list, tel_pos_tensor, num_tel, img_width, img_length, img_depth, training)
    tf.summary.scalar('training_loss', loss)
    tf.summary.scalar('training_accuracy',accuracy)

    #global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    increment_global_step_op = tf.assign(global_step, global_step+1)

    #variable learning rate
    learning_rate = tf.Variable(args.lr,trainable=False)
    num_tel_tensor = tf.constant(num_tel, dtype=tf.float32)
    mean_num_trig_batch = tf.reduce_mean(tf.reduce_sum(tf.to_float(next_trig_list),1))
    scaling_factor = tf.divide(num_tel_tensor,tf.to_float(mean_num_trig_batch))
    variable_learning_rate = tf.multiply(scaling_factor,learning_rate)

    tf.summary.scalar('variable_learning_rate',variable_learning_rate)
    merged = tf.summary.merge_all()

    if image_summary:

        #locate input and 1st layer filter tensors for visualization
        inputs = tf.get_default_graph().get_tensor_by_name("input_0:0")
        kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'MobilenetV1_0/Conv2d_0/convolution:0')[0]
        activations = tf.get_default_graph().get_tensor_by_name("MobilenetV1_0/Conv2d_0/Relu:0")

        variables = [op.name for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name=='Variable']

        #for n in tf.get_default_graph().as_graph_def().node:
            #print(n.name)

        #for i in variables:
            #print(i)

        inputs_charge_summ_op = tf.summary.image('inputs_charge',tf.slice(inputs,begin=[0,0,0,0],size=[TRAIN_BATCH_SIZE,img_width,img_length,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
        inputs_timing_summ_op = tf.summary.image('inputs_timing',tf.slice(inputs,begin=[0,0,0,1],size=[TRAIN_BATCH_SIZE,img_width,img_length,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
        filter_summ_op = tf.summary.image('filter',tf.slice(tf.transpose(kernel, perm=[3, 0, 1, 2]),begin=[0,0,0,0],size=[96,11,11,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
        activations_summ_op = tf.summary.image('activations',tf.slice(activations,begin=[0,0,0,0],size=[TRAIN_BATCH_SIZE,58,58,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
        
    #train op
    if args.optimizer == 'adadelta':
        train_op = tf.train.AdadeltaOptimizer(learning_rate=variable_learning_rate).minimize(loss)
    elif args.optimizer == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate=variable_learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=0.1,
        use_locking=False,
        name='Adam').minimize(loss)
    else:
        train_op = tf.train.GradientDescentOptimizer(variable_learning_rate).minimize(loss)

    #for embeddings visualization
    if embedding:
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
    with sv.managed_session() as sess:
        for i in range(epochs):
            sess.run(training_init_op)
            print("Epoch {} started...".format(i+1))
            while True:
                try:
                    __, __, summ = sess.run([train_op, 
                        increment_global_step_op, merged], 
                        feed_dict={training: True})
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
                    val_loss,val_acc = sess.run([loss,accuracy],feed_dict={training: False})
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

            if i % EPOCHS_PER_IMAGE_VIZ == 0 and image_summary: 
                sess.run(validation_init_op)
                #filter_summ,inputs_summ,activations_summ = sess.run([filter_summ_op,inputs_charge_summ_op,activations_summ_op])
                inputs_summ = sess.run([input_charge_summ_op])
                #sv.summary_computed(sess,filter_summ)
                sv.summary_computed(sess,inputs_summ)
                #sv.summary_computed(sess,activations_summ)

                print("Image summary complete")

            if i % EPOCHS_PER_VIZ_EMBED == 0 and embedding:
                sess.run(validation_init_op)
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
                for k in range(NUM_BATCHES_EMBEDDING):
                    metadata_file.write('{}\n'.format(table_val.read(k,k+1,field=label_column_name)[0]))         
                metadata_file.close()
                
                print("Embedding summary complete")

if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Trains on an hdf5 file.')
    parser.add_argument('h5_file', help='path to h5 file containing data')
    parser.add_argument('--optimizer',default='adam')
    parser.add_argument('--epochs',default=10000,type=int)
    parser.add_argument('--logdir',default='/data0/logs/variable_input_model_1')
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--label_col_name',default='gamma_hadron_label')
    parser.add_argument('--checkpoint_basename',default='custom_multi_input.ckpt')
    parser.add_argument('--embedding', action='store_true')
    parser.add_argument('--no_val',action='store_true')
    parser.add_argument('--image_summary',action='store_true')
    args = parser.parse_args()

    train(variable_input_model,args.h5_file,args.epochs,args.image_summary,args.embedding)
