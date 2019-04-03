import yaml
import operator
from hyperopt import hp, STATUS_OK
from timeit import default_timer as timer
import csv
import pickle
import common

# update ctlearn config file with new hyperparameters


def modify_optimizable_params(self, params):

    iteration = self.iteration

    with open(self.ctlearn_config, 'r') as file:
        myconfig = yaml.load(file)

    stream = self.dependent_hyperparameters

    if stream is not None:
        stream_dict = {}

        ops = {"+": operator.add,
               "-": operator.sub,
               "*": operator.mul,
               "/": operator.truediv}

        for key in stream:
            stream_dict.update({key: ops[stream[key]['operator']](
                stream[key]['factor'], params[stream[key]['hyperparameter']])})

        # add dependent hyperparameters
        params.update(stream_dict)

    myconfig['Logging']['model_directory'] = './run' + str(iteration)
    myconfig['Prediction']['prediction_file_path'] = './run' + \
        str(iteration) + '/predictions_run{}.csv'.format(iteration)

    # update hyperparameters
    common.auxiliar_modify_params(myconfig, params)

    with open(self.ctlearn_config, 'w') as file:
        yaml.dump(myconfig, file)

# create hyperopt style hyperparameters space from optimization config file


def create_space_params(self):

    stream = self.to_be_optimized_hyperparameters
    params = {}
    for key in stream:
        if stream[key]['type'] == 'uniform':
            params.update(
                {key: hp.uniform(key, stream[key]['range'][0],
                                 stream[key]['range'][1])})
        elif stream[key]['type'] == 'quniform':
            params.update(
                {key: hp.quniform(key, stream[key]['range'][0],
                                  stream[key]['range'][1],
                                  stream[key]['step'])})
        elif stream[key]['type'] == 'loguniform':
            params.update(
                {key: 10**hp.uniform(key, stream[key]['range'][0],
                                     stream[key]['range'][1])})
        elif stream[key]['type'] == 'qloguniform':
            params.update(
                {key: 10**hp.quniform(key, stream[key]['range'][0],
                                      stream[key]['range'][1],
                                      stream[key]['step'])})
        elif stream[key]['type'] == 'normal':
            params.update(
                {key: hp.normal(key, stream[key]['range'][0],
                                stream[key]['range'][1])})
        elif stream[key]['type'] == 'qnormal':
            params.update(
                {key: hp.qnormal(key, stream[key]['range'][0],
                                 stream[key]['range'][1],
                                 stream[key]['step'])})
        elif stream[key]['type'] == 'lognormal':
            params.update(
                {key: 10**hp.normal(key, stream[key]['range'][0],
                                    stream[key]['range'][1])})
        elif stream[key]['type'] == 'qnormal':
            params.update(
                {key: 10**hp.qnormal(key, stream[key]['range'][0],
                                     stream[key]['range'][1],
                                     stream[key]['step'])})
        elif stream[key]['type'] == 'choice':
            stream_list = []
            for i in range(len(stream[key]['range'])):
                if not isinstance(stream[key]['range'][i], list):
                    stream_dict = {key: stream[key]['range'][i]}
                    stream_list.append(stream_dict)
                else:
                    stream_dict = {}
                    for j in range(len(stream[key]['range'][i])):

                        if not isinstance(stream[key]['range'][i][j], dict):
                            stream_dict.update(
                                {key: stream[key]['range'][i][j]})
                            stream_list.append(stream_dict)

                        else:
                            for st_key in stream[key]['range'][i][j]:
                                st_type = (stream[key]['range'][i][j][st_key]
                                           ['type'])
                                st_range = (stream[key]['range'][i][j][st_key]
                                            ['range'])
                                if 'step' in (stream[key]['range'][i][j]
                                              [st_key]):
                                    st_step = (stream[key]['range'][i][j]
                                               [st_key]['step'])

                                if st_type == 'uniform':
                                    stream_dict.update({st_key: hp.uniform(
                                        st_key, float(st_range[0]),
                                        float(st_range[1]))})
                                elif st_type == 'quniform':
                                    stream_dict.update({st_key: hp.quniform(
                                        st_key, float(st_range[0]),
                                        float(st_range[1]), float(st_step))})
                                elif st_type == 'loguniform':
                                    stream_dict.update({st_key: 10**hp.uniform(
                                        st_key, float(st_range[0]),
                                        float(st_range[1]))})
                                elif st_type == 'qloguniform':
                                    stream_dict.update(
                                        {st_key: 10**hp.quniform(st_key, float(
                                            st_range[0]), float(st_range[1]),
                                            float(st_step))})
                                elif st_type == 'normal':
                                    stream_dict.update({st_key: hp.normal(
                                        st_key, float(st_range[0]),
                                        float(st_range[1]))})
                                elif st_type == 'qnormal':
                                    stream_dict.update({st_key: hp.qnormal(
                                        st_key, float(st_range[0]),
                                        float(st_range[1]), float(st_step))})
                                elif st_type == 'lognormal':
                                    stream_dict.update({st_key: 10**hp.normal(
                                        st_key, float(st_range[0]),
                                        float(st_range[1]))})
                                elif st_type == 'qlognormal':
                                    stream_dict.update({st_key: 10**hp.qnormal(
                                        st_key, float(st_range[0]),
                                        float(st_range[1]), float(st_step))})
                                elif st_type == 'choice':
                                    raise ValueError('Nested choices are not \
                                    allowed')
                            stream_list.append(stream_dict)
            params.update({key: hp.choice(key, stream_list)})

    return params

# set objective function for hyperopt input - output workflow


def objective(self, params):

    self.iteration += 1
    self.counter += 1

    print('Iteration:', self.counter)
    print('Global iteration:', self.iteration)

    # update hyperparameters
    modify_optimizable_params(self, params)

    start = timer()

    # train ctlearn network
    print('Training')
    common.train(self)
    print('Training ended')
    run_time = timer() - start

    # get validtaion set metrics
    metrics_val = common.get_val_metrics(self)

    # predict if it is set to True
    if self.predict_bool:

        print('Predicting')
        common.predict(self)
        print('Prediction ended')
        metrics_pred = common.get_pred_metrics(self)

        # set loss depending on metric and data set to be optimized
        if self.data_set_to_optimize == 'Validation':
            metric = 'val_' + self.to_be_optimized_metric
            loss = 1 - metrics_val[metric]
            print(metric, ':', metrics_val[metric])

        elif self.data_set_to_optimize == 'Prediction':
            metric = 'pred_' + self.to_be_optimized_metric
            loss = 1 - metrics_pred[metric]
            print(metric, ':', metrics_pred[metric])

        # write hyperparameters and metrics in the checking_file
        with open('./checking_file.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss, self.iteration, params, metrics_val,
                             metrics_pred, run_time])
    else:

        # set loss
        metric = 'val_' + self.to_be_optimized_metric
        loss = 1 - metrics_val[metric]
        print(metric, ':', metrics_val[metric])

        # write hyperparameters and metrics in the checking_file
        with open('./checking_file.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss, self.iteration, params, metrics_val,
                             run_time])

    # save checking_trials file
    pickle.dump(self.trials, open("checking_trials.pkl", "wb"))

    if self.predict_bool:

        return {'loss': loss, 'iteration': self.iteration, 'params': params,
                'metrics_val': metrics_val, 'metrics_pred': metrics_pred,
                'train_time': run_time, 'status': STATUS_OK}
    else:

        return {'loss': loss, 'iteration': self.iteration, 'params': params,
                'metrics_val': metrics_val, 'train_time': run_time,
                'status': STATUS_OK}
