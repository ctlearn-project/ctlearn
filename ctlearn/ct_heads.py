import tensorflow as tf

def particletype_head(cnn_output, logits, num_classes=2):
    logit_units = 1 if num_classes == 2 else num_classes
    prediction_gammahadron_classification = tf.layers.dense(cnn_output, units=logit_units)
    logits.update({'particletype': prediction_gammahadron_classification})
           
    if num_classes == 2:
        head = tf.contrib.estimator.binary_classification_head(name='particletype')
    else:
        head = tf.contrib.estimator.multi_class_head(name='particletype', n_classes=num_classes)
        
    return head
    
def energy_head(cnn_output, logits):
    logit_units = 1
    prediction_energy_regression = tf.layers.dense(cnn_output, units=logit_units)
    logits.update({'energy': prediction_energy_regression})

    head = tf.contrib.estimator.regression_head(name='energy',label_dimension=logit_units)
   
    return head
    
def direction_head(cnn_output, logits):
    logit_units = 2
    prediction_direction_regression = tf.layers.dense(cnn_output, units=logit_units)
    logits.update({'direction': prediction_direction_regression})
    
    head = tf.contrib.estimator.regression_head(name='direction',label_dimension=logit_units)
    
    return head

def impact_head(cnn_output, logits):
    logit_units = 2
    prediction_impact_regression = tf.layers.dense(cnn_output, units=logit_units)
    logits.update({'impact': prediction_impact_regression})
    
    head = tf.contrib.estimator.regression_head(name='impact',label_dimension=logit_units)
    
    return head
    
def showermaximum_head(cnn_output, logits):
    logit_units = 1
    prediction_xmax_regression = tf.layers.dense(cnn_output, units=logit_units)
    logits.update({'showermaximum': prediction_xmax_regression})

    head = tf.contrib.estimator.regression_head(name='showermaximum',label_dimension=logit_units)

    return head
