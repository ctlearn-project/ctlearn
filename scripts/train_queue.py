import argparse
from mvcnn import mvcnn_fn_2
import tensorflow as tf
import re
import numpy as np
import random
import os
import tftables

tf.logging.set_verbosity(tf.logging.DEBUG)

BATCHES_PER_EPOCH = 100

BATCH_SIZE = 32

IMAGE_WIDTH = 240
IMAGE_LENGTH = 240

IMAGE_CHANNELS = 3

STEPS_PER_SUMMARY = 50

STEPS_PER_VALIDATION = 117

STEPS_PER_IMAGE_VIZ = 1000

NUM_IMAGES_EMBEDDING = 700

STEPS_PER_VIZ_EMBED = 1000

STEPS_TO_VIZ_EMBED = 12000

IMAGE_VIZ_MAX_OUTPUTS = 100

def load_data(tbl_batch):

    labels = tf.cast(tbl_batch[args.label_col_name],tf.int8)

    tels_list = [i for i,j in tbl_batch.items() if re.match("T[0-9]+",i)]

    data_list = [tf.to_float(tf.expand_dims(tbl_batch[i],1)) for i in tels_list]

    data_batch = tf.concat(data_list,1)

    return labels, data_batch

def train(data_file,epochs):

    loader_train = tftables.load_dataset(filename=data_file,dataset_path='/E0/Events_Training',input_transform=load_data,batch_size=BATCH_SIZE)
    loader_val = tftables.load_dataset(filename=data_file,dataset_path='/E0/Events_Validation',input_transform=load_data,batch_size=BATCH_SIZE)

    labels_placeholder, images_placeholder = loader_train.dequeue()
    labels_placeholder_val,images_placeholder_val = loader_val.dequeue()

    loss,accuracy,logits,predictions = mvcnn_fn_2(images_placeholder,labels_placeholder)

    #val_loss,val_accuracy,val_logits,val_predictions = mvcnn_fn_2(images_placeholder_val,labels_placeholder_val)

    tf.summary.scalar('training_loss', loss)
    tf.summary.scalar('training_accuracy',accuracy)
    merged = tf.summary.merge_all()

    #val_loss_op = tf.summary.scalar('validation_loss', val_loss)
    #val_acc_op = tf.summary.scalar('validation_accuracy', val_accuracy)

    #locate input and 1st layer filter tensors for visualization
    inputs = tf.get_default_graph().get_tensor_by_name("Conv_block/input:0")
    kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'Conv_block/conv1/kernel:0')[0]
    activations = tf.get_default_graph().get_tensor_by_name("Conv_block/conv1/BiasAdd:0")

    inputs_charge_summ_op = tf.summary.image('inputs_charge',tf.slice(inputs,begin=[0,0,0,0],size=[BATCH_SIZE,IMAGE_WIDTH,IMAGE_LENGTH,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
    inputs_timing_summ_op = tf.summary.image('inputs_timing',tf.slice(inputs,begin=[0,0,0,1],size=[BATCH_SIZE,IMAGE_WIDTH,IMAGE_LENGTH,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
    filter_summ_op = tf.summary.image('filter',tf.slice(tf.transpose(kernel, perm=[3, 0, 1, 2]),begin=[0,0,0,0],size=[96,11,11,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
    activations_summ_op = tf.summary.image('activations',tf.slice(activations,begin=[0,0,0,0],size=[BATCH_SIZE,58,58,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)

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
        for i in range(BATCHES_PER_EPOCH*epochs):
            if sv.should_stop():
                break
            if i % STEPS_PER_SUMMARY == 0:
                summ = sess.run(merged)
                sv.summary_computed(sess, summ)
            
            if i % STEPS_PER_VALIDATION == 0:                   
                '''
                val_accuracies = []
                val_losses = []
                for n in NUM_VAL_BATCHES:
                    val_loss,val_acc = sess.run([loss,accuracy])
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
                '''
            if i % STEPS_PER_IMAGE_VIZ == 0: 
                filter_summ,inputs_summ,activations_summ = sess.run([filter_summ_op,inputs_charge_summ_op,activations_summ_op])
                sv.summary_computed(sess,filter_summ)
                sv.summary_computed(sess,inputs_summ)
                sv.summary_computed(sess,activations_summ)

                print("Image summary complete")

            if i % STEPS_PER_VIZ_EMBED == 0:

                '''
                #reset embedding variable to empty
                sess.run(reset_embedding)

                for j in range(NUM_IMAGES_EMBEDDING):
                    embed_feed_dict = {}
                    embed_data_batches = []
                    for tel in tels_list:
                        embed_data_batches.append(table_val.read(j,j+1,field=tel))
                    embed_feed_dict[images_placeholder] = np.stack(embed_data_batches,axis=1)
                    embed_feed_dict[labels_placeholder] = table_val.read(j,j+1,field=label_column_name) 
                    
                    result1 = sess.run(fetch)
                    embed_feed_dict[fetch] = result1
                    result2 = sess.run(new_embedding_var)
                    embed_feed_dict[new_embedding_var] = result2
                    result3 = sess.run(update_embedding)

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
                '''

            sess.run([train_op,increment_global_step_op])

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
