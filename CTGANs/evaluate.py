#%%
import numpy as np
from sklearn import metrics
import time
import os
import matplotlib.pyplot as plt


def evaluate_on_real(dataset, particle_type_classifier, energy_regressor, direction_regressor):
    # Create empty lists to store the predictions
    particle_type_predictions = []
    energy_predictions = []
    direction_predictions = []
    # Labels will be stored for future usage
    labels = {
        'particletype': [],
        'energy': [],
        'direction': []
    }
    # Get errors
    for batch in range(dataset.__len__()):
        # Load the real batch
        images, batch_labels = dataset.__getitem__(batch)
        # Save the labels in the dictionary
        for key, value in batch_labels.items():
            labels[key].append(value)

        # Predict particle type
        particle_type_predictions.append(particle_type_classifier.predict(images))
        # Predict energy
        energy_predictions.append(energy_regressor.predict(images))
        # Predict arrival direction
        direction_predictions.append(direction_regressor.predict(images))

    # Turn the stored predictions into numpy arrays (remove batch dimension)
    particle_type_predictions = np.concatenate(particle_type_predictions, axis=0)
    energy_predictions = np.concatenate(energy_predictions, axis=0)
    direction_predictions = np.concatenate(direction_predictions, axis=0)
    # Remove batch dimension from label lists
    for key in labels.keys():
        labels[key] = np.concatenate(labels[key], axis=0)

    # Compute the metrics
    particle_type_acc = metrics.accuracy_score(np.argmax(labels['particletype'], axis=1), np.argmax(particle_type_predictions, axis=1))
    energy_mae = metrics.mean_absolute_error(labels['energy'], energy_predictions)
    direction_mae = metrics.mean_absolute_error(labels['direction'], direction_predictions)

    metrics_dict = {
            'particle_type_acc': particle_type_acc,
            'energy_mae': energy_mae,
            'direction_mae': direction_mae
        }

    return metrics_dict


def evaluate_on_generated(dataset, generator, discriminator, particle_type_classifier, energy_regressor, direction_regressor):
    # Create empty lists to store the predictions for generated data
    particle_type_predictions = []
    energy_predictions = []
    direction_predictions = []
    generation_time = []
    w_distance = []
    # Labels will be stored for future usage
    labels = {
        'particletype': [],
        'energy': [],
        'direction': []
    }
    for batch in range(dataset.__len__()):
        # Load the real batch
        real_imgs, batch_labels = dataset.__getitem__(batch)
        # Save the labels in the dictionary
        for key, value in batch_labels.items():
            labels[key].append(value)

        # Generate images for the same labels and measure the time it takes
        labels_for_generation = {key: value for key, value in batch_labels.items() if key!='particletype'}
        tic = time.time()
        generated_imgs = generator(labels_for_generation)
        toc = time.time()
        generation_time.append((toc-tic)/generated_imgs.shape[0])
        # Predict particle type on generated images
        particle_type_predictions.append(particle_type_classifier.predict(generated_imgs))
        # Predict energy on generated images
        energy_predictions.append(energy_regressor.predict(generated_imgs))
        # Predict arrival direction on generated images
        direction_predictions.append(direction_regressor.predict(generated_imgs))
        # Get W1-distance between real and generated data
        w_distance.append(np.mean(discriminator(generated_imgs))-np.mean(discriminator(real_imgs['images'])))

    # Turn the stored predictions into numpy arrays (remove batch dimension)
    particle_type_predictions = np.concatenate(particle_type_predictions, axis=0)
    energy_predictions = np.concatenate(energy_predictions, axis=0)
    direction_predictions = np.concatenate(direction_predictions, axis=0)
    generation_time = np.mean(generation_time)
    # Remove batch dimension from label lists
    for key in labels.keys():
        labels[key] = np.concatenate(labels[key], axis=0)

    # Compute the metrics
    particle_type_acc = metrics.accuracy_score(np.argmax(labels['particletype'], axis=1), np.argmax(particle_type_predictions, axis=1))
    energy_mae = metrics.mean_absolute_error(labels['energy'], energy_predictions)
    direction_mae = metrics.mean_absolute_error(labels['direction'], direction_predictions)
    w_distance = np.mean(np.array(w_distance))
    log_w_distance = np.log10(np.abs(w_distance))

    metrics_dict = {
            'particle_type_acc': particle_type_acc,
            'energy_mae': energy_mae,
            'direction_mae': direction_mae,
            'w_distance': w_distance,
            'log_w_distance': log_w_distance,
            'generation_time': generation_time
        }

    return metrics_dict


def plot_metrics(metrics_on_real, metrics_on_generated, initial_epoch, total_epochs, plots_dir):
    color = 'darkmagenta'
    epochs = np.linspace(initial_epoch, total_epochs, len(metrics_on_generated['generation_time']))
    # Generation time
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(epochs, metrics_on_generated['generation_time'], color=color)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Generation time per image (s)")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
    fig.savefig(os.path.join(plots_dir, 'generation_time.png'), bbox_inches='tight')

    # Particle type accuray
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(epochs, metrics_on_generated['particle_type_acc'], color=color)
    ax.hlines([metrics_on_real['particle_type_acc']], epochs[0], epochs[-1], color=color, ls='--')
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1])
    fig.savefig(os.path.join(plots_dir, 'accuracy.png'), bbox_inches='tight')

    # Energy MAE
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(epochs, metrics_on_generated['energy_mae'], color=color)
    ax.hlines([metrics_on_real['energy_mae']], epochs[0], epochs[-1], color=color, ls='--')
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Mean absolute error")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
    fig.savefig(os.path.join(plots_dir, 'energy_mae.png'), bbox_inches='tight')

    # Arrival direction MAE
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(epochs, metrics_on_generated['direction_mae'], color=color)
    ax.hlines([metrics_on_real['direction_mae']], epochs[0], epochs[-1], color=color, ls='--')
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Mean absolute error")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
    fig.savefig(os.path.join(plots_dir, 'direction_mae.png'), bbox_inches='tight')

    # Wasserstein distance
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(epochs, metrics_on_generated['w_distance'], color=color)
    ax.hlines([0], epochs[0], epochs[-1], color='black', ls='--')
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Wasserstein-1 distance")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
    fig.savefig(os.path.join(plots_dir, 'w_distance.png'), bbox_inches='tight')

    # Logarithm of (absolute value of) Wasserstein distance
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(epochs, metrics_on_generated['log_w_distance'], color=color)
    ax.hlines([0], epochs[0], epochs[-1], color='black', ls='--')
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Wasserstein-1 distance")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
    fig.savefig(os.path.join(plots_dir, 'log_w_distance.png'), bbox_inches='tight')

    plt.close('all')
