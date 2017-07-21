import argparse
import tftables
from mvcnn import mvcnn_fn_2
import tensorflow as tf
from tables import *
import re
import numpy as np
import random
import os

tf.logging.set_verbosity(tf.logging.DEBUG)

BATCH_SIZE = 32

IMAGE_WIDTH = 240
IMAGE_HEIGHT = 240
IMAGE_CHANNELS = 3

STEPS_PER_SUMMARY = 50

STEPS_PER_VALIDATION = 50

STEPS_PER_IMAGE_VIZ = 1000

NUM_IMAGES_EMBEDDING = 700

STEPS_TO_VIZ_EMBED = 10000

#max validation size, constrained by memory
MAX_VALIDATION_SIZE = 256 

def train(data_file,epochs):
    #open file to determine the telescopes included
    f = open_file(data_file, mode = "r", title = "Input file")
    table = f.root.E0.Events_Training
    table_val = f.root.E0.Events_Validation
    columns_list = table.colnames 
    tels_list = []
    for i in columns_list:
        if re.match("T[0-9]+",i):
            tels_list.append(i)

    num_tel = len(tels_list)

    num_events_training = table.shape[0]
    num_events_val = table_val.shape[0]
    batch_indices = [i for i in range(num_events_training) if i%BATCH_SIZE == 0 and i+BATCH_SIZE < num_events_training]
    batches_per_epoch = len(batch_indices)

    print("Batch size: ",BATCH_SIZE)
    print("Total number of training events: ",num_events_training)
    print("Total number of validation events: ",num_events_val)
    print("Batches per epoch: ",batches_per_epoch) 

    with tf.Graph().as_default():

        #make placeholders
        images_placeholder = tf.placeholder(tf.float32, shape=(None,num_tel,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS))
        labels_placeholder = tf.placeholder(tf.int8, shape=(None))

        loss,accuracy,logits,predictions = mvcnn_fn_2(images_placeholder,labels_placeholder)
 
        tf.summary.scalar('training_loss', loss)
        tf.summary.scalar('training_accuracy',accuracy)
        merged = tf.summary.merge_all()

        val_loss_op = tf.summary.scalar('validation_loss', loss)
        val_acc_op = tf.summary.scalar('validation_accuracy', accuracy)

        #print([x.name for x in tf.global_variables()])

        #print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])

        #locate input and 1st layer filter tensors for visualization
        inputs = tf.get_default_graph().get_tensor_by_name("Conv_block/input:0")

        kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'Conv_block/conv1/kernel:0')[0]

        activations = tf.get_default_graph().get_tensor_by_name("Conv_block/conv1/BiasAdd:0")

        filter_summ_op = tf.summary.image('filter',tf.transpose(kernel, perm=[3, 0, 1, 2]),max_outputs=100)
        inputs_summ_op = tf.summary.image('inputs',tf.slice(inputs,begin=[0,0,0,0],size=[32,240,240,1]),max_outputs=100)
        activations_summ_op = tf.summary.image('activations',tf.slice(activations,begin=[0,0,0,0],size=[32,58,58,1]),max_outputs=100)

        #global step
        global_step = tf.Variable(0, name='global_step', trainable=False)
        increment_global_step_op = tf.assign(global_step, global_step+1)
        
        #train op
        train_op = tf.train.AdadeltaOptimizer(learning_rate=args.lr).minimize(loss)

        #for embeddings visualization
        fetch = tf.get_default_graph().get_tensor_by_name('Classifier/fc7/BiasAdd:0')
        embedding_var = tf.Variable(np.empty((0,4096),dtype=np.float32),name='Embedding_of_fc7',validate_shape=False)
        new_embedding_var = tf.concat([embedding_var,fetch],0)
        update_embedding = tf.assign(embedding_var,new_embedding_var,validate_shape=False)

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
                    data_batches.append(table.read(batch_indices[i%batches_per_epoch],batch_indices[i%batches_per_epoch]+BATCH_SIZE,field=tel))
                feed_dict[images_placeholder] = np.stack(data_batches,axis=1)
                feed_dict[labels_placeholder] = table.read(batch_indices[i%batches_per_epoch],batch_indices[i%batches_per_epoch]+BATCH_SIZE,field='gamma_hadron_label')
                if i% batches_per_epoch == batches_per_epoch-1:
                    random.shuffle(batch_indices)
                if i % STEPS_PER_SUMMARY == 0:
                    summ = sess.run(merged,feed_dict=feed_dict)
                    sv.summary_computed(sess, summ)
                if i % STEPS_PER_VALIDATION == 0:                   
                    val_feed_dict = {}
                    val_data_batches = []
                    start = random.randint(0,num_events_val-MAX_VALIDATION_SIZE-1)
                    for tel in tels_list:
                        val_data_batches.append(table_val.read(start,start + MAX_VALIDATION_SIZE,field=tel))
                    val_feed_dict[images_placeholder] = np.stack(val_data_batches,axis=1)
                    val_feed_dict[labels_placeholder] = table_val.read(start,start + MAX_VALIDATION_SIZE,field='gamma_hadron_label')

                    val_loss_summ,val_acc_summ = sess.run([val_loss_op,val_acc_op],feed_dict=val_feed_dict)
                    sv.summary_computed(sess, val_loss_summ)
                    sv.summary_computed(sess, val_acc_summ)

                if i % STEPS_PER_IMAGE_VIZ == 0: 
                    filter_summ,inputs_summ,activations_summ = sess.run([filter_summ_op,inputs_summ_op,activations_summ_op],feed_dict=feed_dict)
                    sv.summary_computed(sess,filter_summ)
                    sv.summary_computed(sess,inputs_summ)
                    sv.summary_computed(sess,activations_summ)

                if i == STEPS_TO_VIZ_EMBED: 
                    for i in range(NUM_IMAGES_EMBEDDING):
                        embed_feed_dict = {}
                        embed_data_batches = []
                        for tel in tels_list:
                            embed_data_batches.append(table_val.read(i,i+1,field=tel))
                        embed_feed_dict[images_placeholder] = np.stack(embed_data_batches,axis=1)
                        embed_feed_dict[labels_placeholder] = table_val.read(i,i+1,field='gamma_hadron_label') 
                        
                        result1 = sess.run(fetch, feed_dict=embed_feed_dict)
                        embed_feed_dict[fetch] = result1
                        result2 = sess.run(new_embedding_var,feed_dict=embed_feed_dict)
                        embed_feed_dict[new_embedding_var] = result2
                        result3 = sess.run(update_embedding,feed_dict=embed_feed_dict)
                        

                    #print([v.shape for v in tf.global_variables() if v.name == embedding_var.name])
                   
                    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
                    config.model_checkpoint_dir = os.path.abspath(args.logdir)
                    embedding = config.embeddings.add()
                    embedding.tensor_name = embedding_var.name
                    embedding.metadata_path = os.path.abspath(os.path.join(args.logdir, 'metadata.tsv'))
                    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(sv.summary_writer, config) 

                    #create corresponding metadata file
                    metadata_file = open(embedding.metadata_path, 'w')
                    for i in range(NUM_IMAGES_EMBEDDING):
                        metadata_file.write('{}\n'.format(table_val.read(i,i+1,field='gamma_hadron_label')[0]))         
                    metadata_file.close()

                sess.run([train_op,increment_global_step_op],feed_dict=feed_dict)

if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Trains on an hdf5 file.')
    parser.add_argument('h5_file', help='path to h5 file containing data')
    parser.add_argument('--epochs',default=10000)
    parser.add_argument('--logdir',default='runs/mvcnn1')
    parser.add_argument('--lr',default=0.001)
    parser.add_argument('--checkpoint_basename',default='mvcnn.ckpt')
    args = parser.parse_args()

    train(args.h5_file,args.epochs)
