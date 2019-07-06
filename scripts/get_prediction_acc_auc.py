import argparse
import numpy as np
import sklearn.metrics

parser = argparse.ArgumentParser(
    description=("Get acc and auc from the predictions csv file."))
parser.add_argument('predictions_list_file',
        help='list of paths to predictions csv file')
args = parser.parse_args()

# Predictions list has the format: predicted_class,proton,gamma,tel_id,event_number,run_number,class_label
labels = []
predictions = []
gamma_predictions = []
with open(args.predictions_list_file) as f:
    for line in f:
        if not line or line[0] == '#': continue
        predicted_class,proton,gamma,tel_id,event_number,run_number,class_label = line.split(',')
        labels.append(class_label.strip())
        gamma_predictions.append(gamma.strip())
        predictions.append(predicted_class.strip())
    labels = np.array(labels[1:]).astype(np.int)
    gamma_predictions = np.array(gamma_predictions[1:]).astype(np.float)
    predictions = np.array(predictions[1:]).astype(np.int)

fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels,gamma_predictions, pos_label=1)
auc = sklearn.metrics.auc(fpr, tpr)
print("auc = {}".format(auc))

acc = sklearn.metrics.accuracy_score(labels,predictions)
print("acc = {}%".format(acc*100))



'''
labels = tf.convert_to_tensor(labels, dtype=tf.float32)
predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)

#acc, update_op = tf.metrics.accuracy(labels,predictions)

auc, update_op = tf.metrics.auc(labels,predictions)
print(auc)

sess = tf.Session()

result = sess.run(auc)
print(result)

'''
