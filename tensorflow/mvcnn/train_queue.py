import argparse
import tftables
from mvcnn import mvcnn_fn,mvcnn_fn_2
import tensorflow as tf
from tables import *
import re

tf.logging.set_verbosity(tf.logging.DEBUG)

NUM_CLASSES = 1
BATCH_SIZE = 32
QUEUE_SIZE = 64
THREADS = 1

NUM_STEPS = 10

tels_list = ['T5','T6','T7','T8']

def input_transform(tbl_batch):
    labels = tbl_batch['gamma_hadron_label']
    data_batches = []
    for i in tels_list:
        data_batches.append(tbl_batch[i])
 
    tel_batch = tf.cast(tf.stack(data_batches,axis=1),tf.float32)
    labels_batch = tf.cast(labels,tf.uint8)

    return labels_batch,tel_batch

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

    # Open the HDF5 file and create a loader for a dataset.
    # The batch_size defines the length (in the outer dimension)
    # of the elements (batches) returned by the reader.
    # Takes a function as input that pre-processes the data.
    loader = tftables.load_dataset(filename=data_file,
    dataset_path='/E0/Events_Training',
    input_transform=input_transform,
    batch_size=BATCH_SIZE)

    # To get the data, we dequeue it from the loader.
    # Tensorflow tensors are returned in the same order as input_transformation
    labels_batch, tel_batch  = loader.dequeue()
  
    loss,logits,prediction = mvcnn_fn_2(tel_batch,labels_batch)

    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    #my_summary_op = tf.summary.merge_all()

    #create supervised session (summary op can be omitted)
    sv = tf.train.Supervisor(
            init_op=tf.global_variables_initializer(),
            logdir=args.logdir,
            checkpoint_basename=args.checkpoint_basename)

    #training loop in session
    with sv.managed_session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        #threads = tf.train.start_queue_runners(sess=sess)
        #start data loader
        #loader.start(sess)
        for step in range(NUM_STEPS):
            if sv.should_stop():
                break
            else:
                sess.run(train_op)

        #loader.stop(sess)

if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Trains on an hdf5 file.')
    parser.add_argument('h5_file', help='path to h5 file containing data')
    parser.add_argument('--logdir',default='runs/')
    parser.add_argument('--checkpoint_basename',default='mvcnn1.ckpt')
    args = parser.parse_args()

    train(args.h5_file)
