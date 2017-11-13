import sys
import os
import argparse

# Disable info and warning messages (not error messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
slim = tf.contrib.slim
import tables
import numpy as np

# Add parent directory to pythonpath to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from models.variable_input_model import variable_input_model

NUM_PARALLEL_CALLS = 12
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64

EPOCHS_PER_IMAGE_VIZ = 5
IMAGE_VIZ_MAX_OUTPUTS = 100
EPOCHS_PER_VIZ_EMBED = 5
NUM_BATCHES_EMBEDDING = 20

SHUFFLE_BUFFER_SIZE = 10000
MAX_STEPS = 10000

def train(model, data_file, epochs, image_summary, embedding):

    def load_HDF5_data(filename, index, metadata, mode='TRAIN'):

        # Read the data at the given table and index from the file
        f = tables.open_file(filename, mode='r')
        if mode == 'TRAIN':
            table = f.root.E0.Events_Training
        elif mode == 'EVAL':
            table = f.root.E0.Events_Validation
        else:
            raise ValueError("Mode must be 'TRAIN' or 'EVAL'")
        record = table.read(index, index + 1)
        
        telescope_ids = metadata['telescope_ids']
        image_indices = record['tel_map'][0]
        telescope_images = []
        for telescope_id, image_index in zip(telescope_ids, image_indices):
            if image_index == -1:
                # Telescope did not trigger. Its outputs will be
                # dropped out, so input is arbitrary. Use an empty
                # array for efficiency.
                telescope_images.append(np.empty(metadata['image_shape']))
            else:
                telescope_table = f.root.E0._f_get_child(telescope_id)
                telescope_images.append(telescope_table[image_index])
        telescope_images = np.stack(telescope_images).astype(np.float32)
        
        # Get binary values indicating whether each telescope triggered
        telescope_triggers = np.array([0 if i < 0 else 1 for i 
            in image_indices], dtype=np.int8)
        
        # Get classification label by converting CORSIKA particle code
        gamma_hadron_label = record['gamma_hadron_label'].astype(np.int8)
        if gamma_hadron_label[0] == 0: # gamma ray
            gamma_hadron_label[0] = 1
        elif gamma_hadron_label[0] == 101: # proton
            gamma_hadron_label[0] = 0
        
        f.close()
        
        return [telescope_images, telescope_triggers, gamma_hadron_label]

    def load_HDF5_auxiliary_data(filename):
        
        f = tables.open_file(filename, mode='r')
        telescope_positions = []
        for row in f.root.Tel_Table.iterrows():
            telescope_positions.append(row["tel_x"])
            telescope_positions.append(row["tel_y"])
        f.close()
        auxiliary_data = {
            'telescope_positions': np.array(telescope_positions, 
                dtype=np.float32)
            }
        return auxiliary_data

    def load_HDF5_metadata(filename):
       
        f = tables.open_file(filename, mode='r')
        num_train_events = f.root.E0.Events_Training.shape[0]
        num_eval_events = f.root.E0.Events_Validation.shape[0]
        # List of telescope IDs ordered by mapping index
        telescope_ids = ["T" + str(row["tel_id"]) for row 
                in f.root.Tel_Table.iterrows()]
        num_telescopes = f.root.Tel_Table.shape[0]
        # All telescope images have the same shape
        image_shape = f.root.E0._f_get_child(telescope_ids[0]).shape[1:]
        f.close()
        metadata = {
                'num_train_events': num_train_events,
                'num_eval_events': num_eval_events,
                'telescope_ids': telescope_ids,
                'num_telescopes': num_telescopes,
                'image_shape': image_shape
                }
        return metadata

    # TODO: rename this argument
    model_dir = args.logdir

    # Define data loading functions
    load_data = load_HDF5_data
    load_auxiliary_data = load_HDF5_auxiliary_data
    load_metadata = load_HDF5_metadata

    # Get information about the dataset
    metadata = load_metadata(data_file)
    
    # Define model hyperparameters
    hyperparameters = {
            'base_learning_rate': args.lr
            }
    
    # Merge dictionaries for passing to the model function
    params = {**metadata, **hyperparameters}
    
    # Get the auxiliary input (same for every event)
    auxiliary_data = load_auxiliary_data(data_file)
   
    # Create training and evaluation datasets
    def load_train_data(index):
        return load_data(data_file, index, metadata, mode='TRAIN')
    
    def load_eval_data(index):
        return load_data(data_file, index, metadata, mode='EVAL')

    train_dataset = tf.data.Dataset.range(metadata['num_train_events'])
    train_dataset = train_dataset.map(lambda index: tuple(tf.py_func(
                load_train_data,
                [index], 
                [tf.float32, tf.int8, tf.int8])),
            num_parallel_calls=NUM_PARALLEL_CALLS)
    train_dataset = train_dataset.batch(TRAIN_BATCH_SIZE)

    eval_dataset = tf.data.Dataset.range(metadata['num_eval_events'])
    eval_dataset = eval_dataset.map(lambda index: tuple(tf.py_func(
                load_eval_data,
                [index],
                [tf.float32, tf.int8, tf.int8])), 
            num_parallel_calls=NUM_PARALLEL_CALLS)
    eval_dataset = eval_dataset.batch(EVAL_BATCH_SIZE)

    # Define the input functions
    def input_fn(dataset, auxiliary_data, shuffle_buffer_size=None):
        # Get batches of data
        if shuffle_buffer_size:
            dataset = dataset.shuffle(shuffle_buffer_size)
        iterator = dataset.make_one_shot_iterator()
        (telescope_data, telescope_triggers, 
                gamma_hadron_labels) = iterator.get_next()
        # Convert auxiliary data to tensors
        telescope_positions = tf.constant(
                auxiliary_data['telescope_positions'])
        features = {
                'telescope_data': telescope_data, 
                'telescope_triggers': telescope_triggers, 
                'telescope_positions': telescope_positions
                }
        labels = {
                'gamma_hadron_labels': gamma_hadron_labels
                }
        return features, labels
    
    def train_input_fn():
        return input_fn(train_dataset, auxiliary_data, 
                shuffle_buffer_size=SHUFFLE_BUFFER_SIZE)
    
    def eval_input_fn():
        return input_fn(eval_dataset, auxiliary_data, 
                shuffle_buffer_size=None)

    def model_fn(features, labels, mode, params, config):
        
        if (mode == tf.estimator.ModeKeys.TRAIN):
            is_training = True
        else:
            is_training = False
        
        loss, accuracy, logits, predictions = model(features, labels, params,
                is_training)
        
        # Scale the learning rate so batches with fewer triggered
        # telescopes don't have smaller gradients
        trigger_rate = tf.reduce_mean(tf.cast(features['telescope_triggers'], 
            tf.float32))
        # Avoid division by 0
        trigger_rate = tf.maximum(trigger_rate, 0.1)
        scaling_factor = tf.reciprocal(trigger_rate)
        scaled_learning_rate = tf.multiply(scaling_factor, 
                params['base_learning_rate'])
        
        # Define the summaries
        # TODO: calculate accuracy using tf.metrics for consistency
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('scaled_learning_rate', scaled_learning_rate)
        tf.summary.merge_all()
        # Define the evaluation metrics
        eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(
                    tf.cast(labels['gamma_hadron_labels'], tf.float32), 
                    predictions['classes'])
                }
        # Define the train op
        optimizer = tf.train.AdamOptimizer(learning_rate=scaled_learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer)
        
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops)
    
    # Train and evaluate the model
    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir,
            params=params)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    
    print("Training and evaluating...")
    print("Training batch size: ", TRAIN_BATCH_SIZE)
    print("Validation batch size: ", EVAL_BATCH_SIZE)
    print("Training steps per epoch: ", np.ceil(metadata['num_train_events'] 
        / TRAIN_BATCH_SIZE).astype(np.int32))
    print("Total number of training events: ", metadata['num_train_events'])
    print("Total number of validation events: ", metadata['num_eval_events'])
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# Everything below here relates to layer and embedding visualization.
# Ignore it all for now.

    if image_summary:

        #locate input and 1st layer filter tensors for visualization
        inputs = tf.get_default_graph().get_tensor_by_name("input_0:0")
        kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'MobilenetV1_0/Conv2d_0/convolution:0')[0]
        activations = tf.get_default_graph().get_tensor_by_name("MobilenetV1_0/Conv2d_0/Relu:0")

        variables = [op.name for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name=='Variable']

        inputs_charge_summ_op = tf.summary.image('inputs_charge',tf.slice(inputs,begin=[0,0,0,0],size=[TRAIN_BATCH_SIZE,img_width,img_length,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
        inputs_timing_summ_op = tf.summary.image('inputs_timing',tf.slice(inputs,begin=[0,0,0,1],size=[TRAIN_BATCH_SIZE,img_width,img_length,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
        filter_summ_op = tf.summary.image('filter',tf.slice(tf.transpose(kernel, perm=[3, 0, 1, 2]),begin=[0,0,0,0],size=[96,11,11,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS)
        activations_summ_op = tf.summary.image('activations',tf.slice(activations,begin=[0,0,0,0],size=[TRAIN_BATCH_SIZE,58,58,1]),max_outputs=IMAGE_VIZ_MAX_OUTPUTS) 

    #for embeddings visualization
    if embedding:
        fetch = tf.get_default_graph().get_tensor_by_name('Classifier/fc7/BiasAdd:0')
        embedding_var = tf.Variable(np.empty((0,4096),dtype=np.float32),name='Embedding_of_fc7',validate_shape=False)
        new_embedding_var = tf.concat([embedding_var,fetch],0)
        update_embedding = tf.assign(embedding_var,new_embedding_var,validate_shape=False)
        empty = tf.Variable(np.empty((0,4096),dtype=np.float32),validate_shape=False)
        reset_embedding = tf.assign(embedding_var,empty,validate_shape=False)
        
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
