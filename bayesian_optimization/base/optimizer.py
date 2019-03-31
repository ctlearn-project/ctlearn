import bayesian_tpe
import common
import pickle
import hyperopt
import os
import csv
from functools import partial
import yaml


class optimizer:

    def __init__(self, opt_config):

        self.counter = 0
        self.optimization_type = opt_config['optimization_type']
        self.ctlearn_config = opt_config['ctlearn_config']
        self.predict_bool = opt_config['predict']
        self.data_set_to_optimize = opt_config['data_set_to_optimize']
        self.to_be_optimized_metric = opt_config['to_be_optimized_metric']
        self.num_max_evals = opt_config['num_max_evals']
        self.reload_trials = opt_config['reload_trials']
        self.reload_checking_file = opt_config['reload_checking_file']
        self.basic_config = opt_config['Basic_config']
        self.fixed_hyperparameters = opt_config[
            'Hyperparameters']['Fixed_hyperparameters']
        self.dependent_hyperparameters = opt_config[
            'Hyperparameters']['Dependent_hyperparameters']
        self.to_be_optimized_hyperparameters = opt_config[
            'Hyperparameters']['To_be_optimized_hyperparameters']
        if self.data_set_to_optimize == 'Prediction':
            assert(self.predict_bool is True)

        if self.reload_checking_file:

            assert(os.path.isfile('trials.pkl'))

            self.trials = pickle.load(open('trials.pkl', 'rb'))
            print('Found trials.pkl file with {} saved trials'
                  .format(len(self.trials.trials)))
            self.num_max_evals += len(self.trials.trials)
            self.iteration = len(self.trials.trials)

        else:
            self.trials = hyperopt.Trials()
            print('No trials file loaded, starting from scratch')
            self.iteration = 0

    def modify_optimizable_params(self, params):
        if self.optimization_type == 'tree_parzen_estimators':
            bayesian_tpe.modify_optimizable_params(self, params)

    def create_space_params(self):
        if self.optimization_type == 'tree_parzen_estimators':
            params = bayesian_tpe.create_space_params(self)
        return params

    def get_pred_metrics(self):
        return common.get_pred_metrics(self)

    def get_val_metrics(self):
        return common.get_val_metrics(self)

    def set_initial_config(self):
        common.set_initial_config(self)

    def train(self):
        common.train(self)

    def predict(self):
        common.predict(self)

    def objective(self, params):

        if self.optimization_type == 'tree_parzen_estimators':

            return bayesian_tpe.objective(self, params)

    def optimize(self):

        if self.reload_checking_file:

            assert(os.path.isfile('./checking_file.csv'))
            with open('./checking_file.csv', 'r') as file:
                existing_iters_csv = len(file.readlines()) - 1

            print('Found checking_file.csv with {} saved trials, \
                     new trials will be added'.format(existing_iters_csv))

            if existing_iters_csv != self.iteration:
                print('Caution: the number of saved trials in trials.pkl \
                    and checking_file.csv files  does not match')

        else:
            print('No checking_file.csv file loaded, starting from scratch')

            with open('./checking_file.csv', 'w') as file:

                writer = csv.writer(file)

                if self.predict_bool:
                    writer.writerow(
                        ['loss', 'iteration', 'params', 'metrics_val',
                         'metrics_pred', 'run_time'])
                else:
                    writer.writerow(
                        ['loss', 'iteration', 'params', 'metrics_val',
                         'run_time'])

        self.set_initial_config()
        parameter_space = self.create_space_params()

        if self.optimization_type == 'tree_parzen_estimators':
            algo = partial(hyperopt.tpe.suggest, n_startup_jobs=20)
        if self.optimization_type == 'random_search':
            algo = hyperopt.rand.suggest

        fmin = hyperopt.fmin(self.objective, parameter_space,
                             algo, trials=self.trials,
                             max_evals=self.num_max_evals)

        pickle.dump(self.trials, open('trials.pkl', 'wb'))
        print('trials.pkl saved')
        print('Best set of hyperparameter found:', fmin)

        print('Optimization run finished')


if __name__ == "__main__":

    with open('opt_config.yml', 'r') as opt_config:
        opt_config = yaml.load(opt_config)

    model = optimizer(opt_config)
    model.optimize()
