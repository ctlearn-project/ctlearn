import wandb
from evaluate import evaluate_on_generated, evaluate_on_real, plot_metrics
from predictor import get_predictor
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import shutil
import json

from data_generator import plot_grid


class Checkpoint(callbacks.Callback):
    def __init__(
        self,
        config_path,
        dataset,
        particle_type_classifier_config,
        energy_regressor_config,
        direction_regressor_config,
        epochs=1,
        nrows=5,
        ncols=5,
        images_dir='images',
        models_dir='models',
        initial_epoch=0
    ):
        # Create the directories (empty if the training is new) to store the models and the images
        if initial_epoch == 0:
            if os.path.exists(models_dir):
                shutil.rmtree(models_dir)
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)

            os.makedirs(models_dir)
            os.makedirs(images_dir)
        
        # Copy the config file to the models directory
        shutil.copyfile(config_path, os.path.join(models_dir, 'GANs.yml'))

        config_artifact = wandb.Artifact('config_file', type='config_file')
        config_artifact.add_file(config_path)
        wandb.log_artifact(config_artifact)
        # Get the real images to plot them and their labels to generate images with the saved models
        features, labels = dataset.__getitem__(0)
        self.labels = {key: value for key, value in labels.items() if key!='particletype'}
        # Plot a grid of real images to compare with the generated ones afterwards
        plot_grid(features['images'], nrows, ncols, save_path=os.path.join(images_dir, 'real_images.png'))

        self.dataset = dataset
        self.particle_type_classifier = get_predictor(**particle_type_classifier_config)
        self.energy_regressor = get_predictor(**energy_regressor_config)
        self.direction_regressor = get_predictor(**direction_regressor_config)
        self.epochs = epochs
        self.nrows = nrows
        self.ncols = ncols
        self.images_dir = images_dir
        self.models_dir = models_dir
        self.initial_epoch = initial_epoch
        self.losses_file_path = os.path.join(self.models_dir, 'losses.json')
        self.metrics_on_real_file_path = os.path.join(self.models_dir, 'metrics_on_real.json')
        self.metrics_on_generated_file_path = os.path.join(self.models_dir, 'metrics_on_generated.json')
                
        # Load losses if the training is being resumed; otherwise, create an empty dictionary
        if initial_epoch != 0 and os.path.exists(self.losses_file_path):
            with open(self.losses_file_path, 'r') as file:
                self.losses = json.load(file)
        else:
            self.losses = {'g_loss': [], 'd_loss': []}

        # Load metrics for real data if the training is being resumed; otherwise, compute them
        if initial_epoch != 0 and os.path.exists(self.metrics_on_real_file_path):
            with open(self.metrics_on_real_file_path, 'r') as file:
                self.metrics_on_real = json.load(file)
        else:
            self.metrics_on_real = evaluate_on_real(dataset, self.particle_type_classifier, self.energy_regressor, self.direction_regressor)
            with open(self.metrics_on_real_file_path, 'w') as file:
                json.dump(self.metrics_on_real, file)
            
        # Load metrics for generated data if the training is being resumed; otherwise, create an empty dictionary
        if initial_epoch != 0 and os.path.exists(self.metrics_on_generated_file_path):
            with open(self.metrics_on_generated_file_path, 'r') as file:
                self.metrics_on_generated = json.load(file)
        else:
            self.metrics_on_generated = {
                'particle_type_acc': [],
                'energy_mae': [],
                'direction_mae': [],
                'w_distance': [],
                'log_w_distance': [],
                'generation_time': []
            }
       

    def _plot_loss(self, logs={}):
        # Plot g_loss and d_loss
        fig, ax1 = plt.subplots()
        epochs = [epoch+1 for epoch in range(len(logs['d_loss']))]
        ax1.plot(epochs, logs['d_loss'], c='darkcyan', label='$D$ loss')
        ax2 = ax1.twinx()
        ax2.plot(epochs, logs['g_loss'], c='darkmagenta', label='$G$ loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Discriminator loss', c='darkcyan')
        ax2.set_ylabel('Generator loss', c='darkmagenta')
        # fig.legend(loc='upper center', ncol=2)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(os.path.join(self.images_dir, 'losses.png'))


    def _generate_and_save(self, epoch):
        # Save the generator and the discriminator
        generator_path = os.path.join(self.models_dir, f'generator_{epoch}')
        discriminator_path = os.path.join(self.models_dir, f'discriminator_{epoch}')
        self.model.generator.save(generator_path)
        self.model.discriminator.save(discriminator_path)
        # Plot a grid of generated images
        images = self.model.generator(self.labels).numpy()
        images_path = os.path.join(self.images_dir, f'generated_images_{epoch}.png')
        plot_grid(images, self.nrows, self.ncols, images_path)

        wandb.log({'samples': wandb.Image(images_path)}, step=epoch)
        g_artifact = wandb.Artifact('Generator', type='model')
        d_artifact = wandb.Artifact('Discriminator', type='model')
        g_artifact.add_dir(generator_path)
        d_artifact.add_dir(discriminator_path)
        wandb.log_artifact(g_artifact)
        wandb.log_artifact(d_artifact)


    # TODO:
    # def on_train_begin(self, logs=None):
    #     # Print generator summary and plot the model
    #     self.model.generator.summary()
    #     gen_plot_file = os.path.join(self.images_dir, 'generator.png')
    #     tf.keras.utils.plot_model(self.model.generator, show_shapes=True, to_file=gen_plot_file)
    #     # Print discriminator summary and plot the model
    #     self.model.discriminator.summary()
    #     disc_plot_file = os.path.join(self.images_dir, 'discriminator.png')
    #     tf.keras.utils.plot_model(self.model.discriminator, show_shapes=True, to_file=disc_plot_file)


    def on_epoch_end(self, epoch, logs=None):
        # Get the real epoch value if the training has been resumed
        epoch += self.initial_epoch + 1
        # Store the losses of the latest epoch
        for key in ('g_loss', 'd_loss'):
            self.losses[key].append(logs[key])
        
        # Update the loss plots
        self._plot_loss(self.losses)
        # Update the loss file
        with open(self.losses_file_path, 'w') as file:
            json.dump(self.losses, file)
        
        wandb.log({'training': {'epoch': epoch, **logs}}, step=epoch)

        if epoch%self.epochs == 0:
            # Save the models after the specified number of epochs and plot samples
            self._generate_and_save(epoch)
            # Compute the metrics for generated data
            metrics_on_generated = evaluate_on_generated(
                self.dataset,
                self.model.generator,
                self.model.discriminator,
                self.particle_type_classifier,
                self.energy_regressor,
                self.direction_regressor
            )

            wandb.log({'validation': {'epoch': epoch, **metrics_on_generated}}, step=epoch)
            # Update the metrics history
            for key, value in metrics_on_generated.items():
                self.metrics_on_generated[key].append(float(value))

            # Update the metrics file
            with open(self.metrics_on_generated_file_path, 'w') as file:
                json.dump(self.metrics_on_generated, file)
            
            # Plot the metrics
            plot_metrics(self.metrics_on_real, self.metrics_on_generated, self.initial_epoch, self.epochs, self.images_dir)

    
    def on_train_end(self, logs=None):
        plots_artifact = wandb.Artifact('Plots', type='plots')
        plots_artifact.add_file(self.losses_file_path)
        plots_artifact.add_file(self.metrics_on_real_file_path)
        plots_artifact.add_file(self.metrics_on_generated_file_path)
        wandb.log_artifact(plots_artifact)
