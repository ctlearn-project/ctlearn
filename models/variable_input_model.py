import tensorflow as tf
import numpy as np

NUM_CLASSES = 2

IMAGE_WIDTH = 120
IMAGE_LENGTH = 120
IMAGE_DEPTH = 1

NUM_FEATURES = 1024

NUM_TEL = 15

#for use with train_datasets
def alexnet_block(input_features,trig_values,number):

    input_features.set_shape([None,IMAGE_WIDTH,IMAGE_LENGTH,IMAGE_DEPTH])
    #shared weights
    if number == 0:
        reuse = None
    else:
        reuse = True

    with tf.variable_scope("Conv_block"):

        #conv1
        conv1 = tf.layers.conv2d(
                inputs=input_features,
                filters=96,
                kernel_size=[11, 11],
                strides=4,
                padding="valid",
                activation=tf.nn.relu,
                name="conv1",
                reuse=reuse,
                kernel_initializer = tf.zeros_initializer())

        #local response normalization ???

        #pool1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

        #conv2
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=256,
                kernel_size=[5, 5],
                padding="valid",
                activation=tf.nn.relu,
                name="conv2",
                reuse=reuse)

        #normalization ????

        #pool2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

        #conv3
        conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=384,
                kernel_size=[3,3],
                padding="valid",
                activation=tf.nn.relu,
                name="conv3",
                reuse=reuse)

        #conv4
        conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=384,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu,
                name="conv4",
                reuse=reuse)

        #conv5
        conv5 = tf.layers.conv2d(
                inputs=conv4,
                filters=256,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu,
                name="conv5",
                reuse=reuse)

        #pool5
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

        #flatten output of pool5 layer to get feature vector of shape (num_batch,1024)
        dim = np.prod(pool5.get_shape().as_list()[1:])
        reshape = tf.reshape(pool5, [-1, dim])
        output = tf.multiply(reshape,tf.to_float(trig_values))

    return output

#for use with train_datasets
def variable_input_model(tel_data,labels,trig_list,tel_pos_tensor,training):

    #from batch,tel,width,length,channels to tel,batch,width,length,channels
    tel_data_transposed = tf.transpose(tel_data, perm=[1, 0, 2, 3, 4])
    trig_list_transposed = tf.transpose(trig_list, perm=[1,0])

    feature_vectors = []
 
    for i in range(NUM_TEL):
        feature_vectors.append(tf.expand_dims(alexnet_block(tf.gather(tel_data_transposed,i),tf.gather(trig_list_transposed,i),i),1))

    with tf.variable_scope("Classifier"):

        #shape = (num_batch,num_tel,1024)
        combined_feature_tensor = tf.concat(feature_vectors, 1)
        
        batch_size = tf.shape(combined_feature_tensor)[0]
        #tel_pos_tensor is initially shape = (num_tel,2)
        #convert to shape = (1,num_tel)
        tel_pos_tensor_batch = tf.expand_dims(tel_pos_tensor,0)
        #convert to shape = (batch_size,num_tel,2) by tiling along 1st dimension
        tel_pos_tensor_batch =  tf.tile(tel_pos_tensor_batch, tf.stack([batch_size,1,1])) 
        #trig_list initially shape (batch_size,num_tel)
        trig_list.set_shape([None,NUM_TEL])
        #convert to shape = (batch_size,num_tel,1)
        #concatenate all along dimension 2 (1024+2+1)
        combined_feature_tensor = tf.concat([combined_feature_tensor,tf.to_float(tf.expand_dims(trig_list,-1)),tel_pos_tensor_batch],2)
        #flatten
        #combined_feature_vector = tf.reshape(combined_feature_tensor, [batch_size,-1])
        #fc6
        fc6 = tf.layers.dense(inputs=combined_feature_tensor, units=4096, activation=tf.nn.relu,name="fc6") 
        dropout6 = tf.layers.dropout(inputs=fc6, rate=0.5, training=training)

        #fc7
        fc7 = tf.layers.dense(inputs=dropout6, units=4096, activation=tf.nn.relu,name="fc7")        
        dropout7 = tf.layers.dropout(inputs=fc7, rate=0.5, training=training)        

        #fc8
        fc8 = tf.layers.dense(inputs=dropout7, units=NUM_CLASSES,name="fc8")

    with tf.variable_scope("Outputs"):

        # Calculate Loss (for both TRAIN and EVAL modes) 
        if NUM_CLASSES == 2:
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=onehot_labels, logits=fc8)
        else:
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=fc8)

        #outputs
        if NUM_CLASSES == 2:
            predictions = {
                    "classes": tf.argmax(input=fc8,axis=1),
                    "probabilities": [tf.sigmoid(fc8), 1-tf.sigmoid(fc8)]
                    }
        else:
            predictions = {        
                    "classes": tf.argmax(input=fc8, axis=1),
                    "probabilities": tf.nn.softmax(fc8, name="softmax_tensor")
                    }

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predictions['classes'],tf.int8),labels), tf.float32))

    return loss,accuracy,fc8,predictions


