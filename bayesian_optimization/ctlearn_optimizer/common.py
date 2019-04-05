#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sklearn
import os
import re
import yaml
from ctlearn.run_model import run_model
from multiprocessing import Pool

# auxiliar function to modify ctlearn config hyperparameters
def auxiliar_modify_params(myconfig, params):

    if 'layer1_filters' in params:
        myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'][0]['filters'] = int(params['layer1_filters'])
    if 'layer1_kernel' in params:
        myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'][0]['kernel_size'] = int(params['layer1_kernel'])
    if 'layer2_filters' in params:
        myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'][1]['filters'] = int(params['layer2_filters'])
    if 'layer2_kernel' in params:
        myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'][1]['kernel_size'] = int(params['layer2_kernel'])
    if 'layer3_filters' in params:
        myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'][2]['filters'] = int(params['layer3_filters'])
    if 'layer3_kernel' in params:
        myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'][2]['kernel_size'] = int(params['layer3_kernel'])
    if 'layer4_filters' in params:
        myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'][3]['filters'] = int(params['layer4_filters'])
    if 'layer4_kernel' in params:
        myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'][3]['kernel_size'] = int(params['layer4_kernel'])
    if 'pool_size' in params:
        myconfig['Model']['Model Parameters']['basic']['conv_block']['max_pool']['size'] = int(params['pool_size'])
    if 'pool_strides' in params:
        myconfig['Model']['Model Parameters']['basic']['conv_block']['max_pool']['strides'] = int(params['pool_strides'])
    if 'optimizer_type' in params:
        if params['optimizer_type'] is dict:
            myconfig['Training']['Hyperparameters']['optimizer'] = params['optimizer_type']['optimizer_type']
        else:
            myconfig['Training']['Hyperparameters']['optimizer'] = params['optimizer_type']
    if 'base_learning_rate' in params:
        myconfig['Training']['Hyperparameters']['base_learning_rate'] = params['base_learning_rate']
    if 'adam_epsilon' in params:
        myconfig['Training']['Hyperparameters']['adam_epsilon'] = params['adam_epsilon']
    if 'cnn_rnn_dropout' in params:
        myconfig['Model']['Model Parameters']['cnn_rnn']['dropout_rate'] = params['cnn_rnn_dropout']

# get prediction set metrics
def get_pred_metrics(self):

    #load prediction.csv and compute metrics
    predictions_path = './run' + \
        str(self.iteration) + '/predictions_run{}.csv'.format(self.iteration)

    predictions = np.genfromtxt(predictions_path, delimiter=',', names=True)
    labels = predictions['gamma_hadron_label'].astype(int)
    gamma_classifier_values = predictions['gamma']
    predicted_class = predictions['predicted_class'].astype(int)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        labels, gamma_classifier_values, pos_label=0)
    auc = sklearn.metrics.auc(fpr, tpr)
    f1 = sklearn.metrics.f1_score(labels, predicted_class)
    acc = sklearn.metrics.accuracy_score(labels, predicted_class)
    bacc = sklearn.metrics.balanced_accuracy_score(labels, predicted_class)
    prec = sklearn.metrics.precision_score(labels, predicted_class)
    rec = sklearn.metrics.recall_score(labels, predicted_class)
    log_loss = sklearn.metrics.log_loss(labels, predicted_class)

    metrics_pred = {'pred_auc': auc, 'pred_acc': acc, 'pred_bacc': bacc,
                    'pred_f1': f1, 'pred_prec': prec, 'pred_rec': rec,
                    'pred_log_loss': log_loss}

    return metrics_pred

# get validation set metrics
def get_val_metrics(self):

    iteration = self.iteration

    #load training log file and get validation metrics
    run_folder = './run' + str(iteration)
    for file in os.listdir(run_folder):
        if file.endswith('logfile.log'):
            with open(run_folder + '/' + file) as log_file:
                contents = log_file.read()

                if 'Training' in contents:
                    train_logfile = file

    with open(run_folder + '/' + train_logfile, 'r') as stream:
        r = re.compile('INFO:Saving dict for global step .*')
        matches = list(filter(r.match, stream))
        assert(len(matches) > 0)
        val_info = matches[-1]

    auc = float(re.findall(r'auc = [-+]?\d*\.*\d+', val_info)[0][6:])
    accuracy = float(re.findall(r'accuracy = [-+]?\d*\.*\d+', val_info)[0][11:])
    accuracy_gamma = float(re.findall(r'accuracy_gamma = [-+]?\d*\.*\d+', val_info)[0][17:])
    accuracy_proton = float(re.findall(r'accuracy_proton = [-+]?\d*\.*\d+', val_info)[0][18:])
    loss = float(re.findall(r'loss = [-+]?\d*\.*\d+', val_info)[0][7:])

    metrics_val = {'val_auc': auc, 'val_acc': accuracy, 'val_acc_gamma':
                   accuracy_gamma, 'val_acc_proton': accuracy_proton,
                   'val_loss': loss}

    return metrics_val

# set basic config and not optimizable hyperparameters
def set_initial_config(self):

    with open(self.ctlearn_config, 'r') as myconfig:
        myconfig = yaml.load(myconfig)

    myconfig['Training']['num_validations'] = self.basic_config['num_validations']
    myconfig['Training']['num_training_steps_per_validation'] = self.basic_config['num_training_steps_per_validation']
    myconfig['Data']['Loading']['example_type'] = self.basic_config['example_type']
    myconfig['Data']['Input']['batch_size'] = self.basic_config['batch_size']
    myconfig['Model']['model_directory'] = self.basic_config['model_directory']
    myconfig['Data']['Loading']['validation_split'] = self.basic_config['validation_split']

    if self.basic_config['example_type'] == 'array':
        myconfig['Data']['Loading']['merge_tel_types'] = True
        myconfig['Model']['model']['module'] = 'cnn_rnn'
        myconfig['Model']['model']['function'] = 'cnn_rnn_model'

    elif self.basic_config['example_type'] == 'single_tel':
        myconfig['Data']['Loading']['merge_tel_types'] = True
        myconfig['Model']['model']['module'] = 'single_tel'
        myconfig['Model']['model']['function'] = 'single_tel_model'

    myconfig['Data']['Loading']['selected_tel_types'] = self.basic_config['selected_tel_types']

    #make sure that the number of convolutional layers is correct
    if self.basic_config['selected_tel_types'] == (['SST:ASTRICam'] or ['SST:CHEC']):

        while len(myconfig['Model']['Model Parameters']['basic']['conv_block']['layers']) > 3:
            del myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'][-1]
        while len(myconfig['Model']['Model Parameters']['basic']['conv_block']['layers']) < 3:
            myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'].append({'filters': 288, 'kernel_size': 288})

    else:

        while len(myconfig['Model']['Model Parameters']['basic']['conv_block']['layers']) > 4:
            del myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'][-1]
        while len(myconfig['Model']['Model Parameters']['basic']['conv_block']['layers']) < 4:
            myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'].append({'filters': 288, 'kernel_size': 288})

    params = self.fixed_hyperparameters

    #modify ctlearn config file
    auxiliar_modify_params(myconfig, params)

    with open(self.ctlearn_config, 'w') as file:
        yaml.dump(myconfig, file)


def run_train(config):

    run_model(config, mode='train', debug=False, log_to_file=True)

# run training
def train(self):

    with open(self.ctlearn_config, 'r') as file:
        myconfig = yaml.load(file)

    myconfig['Data']['file_list'] = self.basic_config['training_file_list']

    with open(self.ctlearn_config, 'w') as file:
        yaml.dump(myconfig, file)

    with open(self.ctlearn_config, 'r') as myconfig:
        config = yaml.load(myconfig)

    with Pool(1) as p:
        p.apply(run_train, (config,))


def run_pred(config):

    run_model(config, mode='predict', debug=False, log_to_file=True)

# run prediction
def predict(self):

    with open(self.ctlearn_config, 'r') as file:
        myconfig = yaml.load(file)

    myconfig['Data']['file_list'] = self.basic_config['prediction_file_list']

    with open('myconfig.yml', 'w') as file:
        yaml.dump(myconfig, file)

    with open('myconfig.yml', 'r') as myconfig:
        config = yaml.load(myconfig)

    with Pool(1) as p:
        p.apply(run_pred, (config,))
