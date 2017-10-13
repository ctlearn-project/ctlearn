import os
import struct

import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.data import Dataset, Iterator

from mobilenet_v1 import mobilenet_v1, Conv, DepthSepConv

BATCH_SIZE = 64
NUM_TRAINING_STEPS = 4000
DATA_PATH = '/home/shevek/brill/mobilenet/MNIST-data'
CHECKPOINT_DIR = '/home/shevek/brill/mobilenet/checkpoints'

# Read in MNIST dataset
# https://gist.github.com/akesling/5358964
def read_mnist(dataset="training", path="."):
    """
    Python function for importing the MNIST data set. It returns an iterator
    of tuples of tensors where the first entries are the labels and the second 
    are the images.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.uint8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        channels = 1
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols, 
                channels)

    # Convert to tensors
    images = tf.convert_to_tensor(img, dtype=tf.float32)
    labels = tf.convert_to_tensor(lbl, dtype=tf.int32)

    return images, labels

def get_batch(dataset, batch_size=BATCH_SIZE, data_path=DATA_PATH):
    """
    dataset can be 'training' or 'testing'
    """
    # Create dataset
    mnist_images, mnist_labels = read_mnist(dataset, path=data_path)
    mnist_data = Dataset.from_tensor_slices((mnist_images, mnist_labels))
    mnist_data = mnist_data.repeat()
    mnist_data = mnist_data.batch(batch_size)
    
    # Create iterator
    iterator = mnist_data.make_one_shot_iterator()
    images, labels = iterator.get_next()
    
    return images, labels

# Define custom MobileNet for MNIST
conv_defs = [
    Conv(kernel=[3, 3], stride=1, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]

graph = tf.Graph()
with graph.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # Get a batch of data
    training_images, training_labels = get_batch('training')
    validation_images, validation_labels = get_batch('testing')
    
    # Create model for training and validation
    with tf.variable_scope("model") as scope:
        training_logits, training_end_points = mobilenet_v1(
                training_images, num_classes=10, dropout_keep_prob=1.0,
                conv_defs=conv_defs)
        scope.reuse_variables()
        validation_logits, end_points_validation = mobilenet_v1(
                validation_images, num_classes=10, dropout_keep_prob=1.0, 
                conv_defs=conv_defs)
    
    # Complete the graph
    training_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=training_labels, logits=training_logits)
    validation_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=validation_labels, logits=validation_logits)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    train_op = slim.learning.create_train_op(training_loss, optimizer)
    
    # Create some summaries to visualize the training process
    tf.summary.scalar('losses/Training_Loss', training_loss)
    tf.summary.scalar('losses/Validation_Loss', validation_loss)
    training_accuracy = slim.metrics.accuracy(tf.argmax(training_logits,
        axis=1, output_type=tf.int32), training_labels)
    tf.summary.scalar('metrics/Training_Accuracy', training_accuracy)
    validation_accuracy = slim.metrics.accuracy(tf.argmax(validation_logits,
        axis=1, output_type=tf.int32), validation_labels)
    tf.summary.scalar('metrics/Validation_Accuracy', validation_accuracy)

# Train
final_loss = slim.learning.train(
        train_op,
        logdir=CHECKPOINT_DIR,
        number_of_steps=NUM_TRAINING_STEPS,
        graph=graph,
        save_summaries_secs=10,
        log_every_n_steps=NUM_TRAINING_STEPS)

print("Finished training. Last batch loss:", final_loss)
