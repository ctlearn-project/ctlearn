import argparse
import tftables
from mvcnn import mvcnn_fn,mvcnn_fn_2
import tensorflow as tf
from tables import *
import re

tf.logging.set_verbosity(tf.logging.DEBUG)

NUM_CLASSES = 1
BATCH_SIZE = 8
QUEUE_SIZE = 32
THREADS = 1

NUM_STEPS = 10

def train(data_file):

    #open file to determine the telescopes included
    f = open_file(data_file, mode = "r", title = "Input file")
    table = f.root.E0.Events_Training
    columns_list = table.colnames 
    tels_list = []
    for i in columns_list:
        if re.match("T[0-9]+",i):
            tels_list.append(i)
    f.close()

    tels_list = ['T5','T6','T7','T8']

    #prepare reader/loader
    reader = tftables.open_file(filename=data_file, batch_size=BATCH_SIZE)

    #hardcoded to first energy bin
    array_batch_placeholder = reader.get_batch(
            path = '/E0/Events_Training',
            cyclic = True,
            ordered = False)
   
    with tf.device('/cpu:0'):
        
        label_batch = [array_batch_placeholder['gamma_hadron_label']]
        data_batches = []
        for i in tels_list:
            data_batches.append(array_batch_placeholder[i])
     
        inputs_list = data_batches + label_batch

        loader = reader.get_fifoloader(queue_size = QUEUE_SIZE,inputs = inputs_list,threads = THREADS)
       
        batch = loader.dequeue()

        inputs_batch = [tf.cast(i,tf.float32) for i in batch[0:len(batch)-1]]
        labels_batch = tf.cast(batch[len(batch)-1],tf.uint8)

    loss,logits,prediction = mvcnn_fn(inputs_batch,labels_batch)

    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    #my_summary_op = tf.summary.merge_all()

    #create supervised session (summary op can be omitted)
    sv = tf.train.Supervisor(
            init_op=tf.global_variables_initializer(),
            logdir=args.logdir,
            checkpoint_basename=args.checkpoint_basename)

    #training loop in session
    with sv.managed_session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        threads = tf.train.start_queue_runners(sess=sess)
        #start data loader
        loader.start(sess)
        for step in range(NUM_STEPS):
            if sv.should_stop():
                print('Stopping...')
                break
            else:
                sess.run(train_op)

        loader.stop(sess)

if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Trains on an hdf5 file.')
    parser.add_argument('h5_file', help='path to h5 file containing data')
    parser.add_argument('--logdir',default='runs/')
    parser.add_argument('--checkpoint_basename',default='mvcnn1.ckpt')
    args = parser.parse_args()

    train(args.h5_file)
