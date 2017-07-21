
import tftables
from mvcnn import mvcnn_estimator_fn
import tensorflow as tf
import tables

tf.logging.set_verbosity(tf.logging.INFO)

NUM_CLASSES = 1
BATCH_SIZE = 32
QUEUE_SIZE = 10
THREADS = 4

def train(data_file):

    #open file to determine the telescopes included
    f = open_file(data_file, mode = "r", title = "Input file")
    table = h5file.root.E0.Events_Training
    columns_list = table.colnames 
    print(columns_list)

    #prepare reader/loader
    reader = tftables.open_file(filename=data_file, batch_size=BATCH_SIZE)

    array_batch_placeholder = reader.get_batch(
            path = '/E0/Events_Training',
            cyclic = True,
            ordered = False)
   
    label_batch = array_batch_placeholder['gamma_hadron_label']
    data_batches = []
    for i in :
 
    inputs_list = data_batches + label_batch

    loader = reader.get_fifoloader(queue_size = QUEUE_SIZE,inputs = inputs_list,threads = THREADS)
   
    batch = loader.dequeue()

    inputs_batch = batch[0:len(batch)-2]
    label_batch = batch[len(batch)-1]

    result = mvcnn_fn(inputs_batch,labels_batch)

    estimator = tf.contrib.learn.Estimator(model_fn=mvcnn_estimator_fn) 

    input_fn = tf.contrib.learn.io.numpy_input_fn

    estimator.fit(input_fn=input_fn, steps=1000)
    # Here we evaluate how well our model did. 
    train_loss = estimator.evaluate(input_fn=input_fn)
    eval_loss = estimator.evaluate(input_fn=eval_input_fn)
    print("train loss: %r"% train_loss)
    print("eval loss: %r"% eval_loss)
    
if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Trains on an hdf5 file.')
    parser.add_argument('h5_file', help='path to h5 file containing data')
    args = parser.parse_args()

    train(args.h5_file)
