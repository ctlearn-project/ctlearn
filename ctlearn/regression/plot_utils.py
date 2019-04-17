from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import PowerNorm
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def compute_bin_gaussian_error(y_true, y_pred, net_name, num_bins=10, plot=True, save_error=False, fig_folder='',
                               do_show=False):
    '''
    Helper function that compute the gaussian fit statistics for a nuber of bins
    :param y_pred: Predicted y (Log Scale)
    :param y_true: True y (Log scale)
    :param num_bin: Number of bins
    :return: bins_mu, bins_sigma, bins_mean_value
    '''

    gaussian = GaussianMixture(n_components=1)
    bins = np.linspace(1, max(y_true), num_bins)

    bins_mu = np.zeros(num_bins - 1)
    bins_sigma = np.zeros(num_bins - 1)
    bins_median_value = np.zeros(num_bins - 1)

    if plot:
        n_row = int(np.sqrt(num_bins - 1))
        n_col = np.ceil((num_bins - 1) / n_row)
        # axs, fig = plt.subplots(n_row, n_col)
        plt.figure(figsize=(15, 15))

    for i in range(num_bins - 1):
        idx_bin = np.logical_and(y_true > bins[i], y_true < bins[i + 1])
        y_bin_true_lin = np.power(10, y_true[idx_bin])
        y_bin_pred_lin = np.power(10, y_pred[idx_bin].flatten())
        error_pure = np.divide((y_bin_pred_lin - y_bin_true_lin), y_bin_true_lin)
        error_subset = error_pure[
            np.logical_and(error_pure < np.percentile(error_pure, 95), error_pure > np.percentile(error_pure, 5))]
        # error = error_subset[:, np.newaxis]  # Add a new axis just for interface with Gaussian Mixture

        # gaussian.fit(error)
        # mu = gaussian.means_
        # sigma = np.sqrt(gaussian.covariances_)

        # Error sigma as collecting 68% of data
        mu = np.percentile(error_pure, 50)
        up = np.percentile(error_pure, 84)  # 100 - (100-68)/2
        low = np.percentile(error_pure, 16)  # (100-68)/2
        sigma = (up - low) / 2

        bins_mu[i] = mu
        bins_sigma[i] = sigma
        bins_median_value[i] = np.sqrt([bins[i] * bins[i + 1]])
        if save_error:
            np.savetxt('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/errors/' + net_name + 'error_bin_' + str(
                bins_median_value[i]) + '.gz', error_pure)
        if plot:
            plt.subplot(n_row, n_col, i + 1)
            # plt.hist(error.flatten(), bins=50, density=False)
            sns.distplot(error_pure, kde=True, rug=True, bins=50)
            mu = mu.flatten()
            sigma = sigma.flatten()
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            plt.plot(x, norm.pdf(x, mu, sigma))
            plt.title('Median Value: ' + "{:.2E}".format(Decimal(np.power(10, bins_median_value[i]))))
            plt.legend(['Fitted Gaussian', 'Histogram of Error'])

    if plot:
        plt.tight_layout()
        plt.savefig(fig_folder + '/' + net_name + '_GaussianErrorDist.png')
        plt.savefig(fig_folder + '/' + net_name + '_GaussianErrorDist.eps')
        if do_show:
            plt.show()

    bins_median_value_lin = np.power(10, bins_median_value)  # Bins back to linear
    return bins_mu, bins_sigma, bins_median_value_lin


def plot_gaussian_error(y_true, y_pred, net_name, fig_folder, num_bins=10, do_show=False, **kwargs):
    ######## PAPER DATA
    cutting_edge_magic_bins = [[47, 75],
                               [75, 119],
                               [119, 189],
                               [189, 299],
                               [299, 475],
                               [475, 753],
                               [753, 1194],
                               [1194, 1892],
                               [1892, 2999],
                               [2999, 4754],
                               [4754, 7535],
                               [7535, 11943],
                               [11943, 18929]]
    cutting_edge_magic_bins_median = []
    for bins in cutting_edge_magic_bins:
        median = np.sqrt(bins[0] * bins[1])
        cutting_edge_magic_bins_median.append(median)
    cutting_edge_magic_bins_median = np.array(cutting_edge_magic_bins_median)

    cutting_edge_magic_bias = np.array(
        [24.6, 7.1, -0.1, -1.5, -2.2, -2.1, -1.4, -1.8, -2.3, -1.7, -2.6, -2.1, -6.7]) * 0.01
    cutting_edge_magic_sigma = np.array(
        [21.8, 19.8, 18.0, 16.8, 15.5, 14.8, 15.4, 16.1, 18.1, 19.6, 21.9, 22.7, 20.7]) * 0.01
    cutting_edge_magic_RMS = np.array([22.5, 20.9, 21.3, 20.49, 20.20, 20.21, 21.3, 21.3, 23.2, 25.1, 26.5, 26.8, 24.4])
    ########

    bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, y_pred, net_name, num_bins,
                                                                        fig_folder=fig_folder, **kwargs)
    fig_width = 9
    plt.figure(figsize=(fig_width, fig_width * 0.618))
    plt.subplot(1, 2, 1)
    plt.semilogx(bins_median_value, bins_mu, '-*g')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_mu), np.mean(bins_mu)], 'r--')
    plt.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_bias, 'r-o')
    plt.grid(which='both')
    plt.legend(['Estimated $\mu$', 'Cutting Edge Technology'])
    plt.xlabel('Bin median value')
    plt.ylabel('$\mu$ of linear prediction error')
    plt.title('$\mu$ distribution for each bin')
    # plt.savefig('pics/bins_mu.jpg')

    plt.subplot(1, 2, 2)
    # plt.figure()
    plt.semilogx(bins_median_value, bins_sigma, '-*')
    plt.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_sigma, '--o')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
    plt.grid(which='both')
    plt.ylabel('$\sigma$ of linear prediction error')
    plt.xlabel('Bin median value')
    plt.title('$\sigma$ distribution for each bin')
    plt.legend(['Estimated $\sigma$', 'Cutting Edge Technology'])
    plt.tight_layout()
    plt.savefig(fig_folder + '/' + net_name + '_bins.png')
    plt.savefig(fig_folder + '/' + net_name + '_bins.eps')
    if do_show:
        plt.show()


def plot_hist2D(y_true, y_pred, net_name, fig_folder, num_bins=10, do_show=True):
    plt.figure()
    plt.hist2d(x=y_true, y=y_pred.flatten(), bins=num_bins, cmap='inferno', norm=PowerNorm(0.65))
    plt.plot([1, 10], [1, 10], 'w-')
    plt.xlabel('True Energy (Log10)')
    plt.ylabel('Predicted Energy (Log10)')
    plt.colorbar()
    plt.title('Regression Performances ' + net_name)
    plt.legend(['Ideal Line'])
    plt.xlim(1.2, 4.5)
    plt.ylim(1.2, 4.5)
    plt.savefig(fig_folder + '/' + net_name + '_hist2D' + '.png')
    plt.savefig(fig_folder + '/' + net_name + '_hist2D' + '.eps')
    if do_show:
        plt.show()
    plt.close()


def bin_data(data, num_bins, bins=None):
    if bins is None:
        bins = np.linspace(np.min(data), np.max(data), num_bins)
    binned_values = np.zeros(data.shape)
    for i, bin in enumerate(bins):
        if i < bins.shape[0] - 1:
            mask = np.logical_and(data >= bins[i], data <= bins[i + 1])
            binned_values[mask] = bin
    return binned_values, bins


def bin_data_mask(data, num_bins, bins=None):
    if bins is None:
        bins = np.linspace(np.min(data), np.max(data), num_bins)
    binned_values = np.zeros(data.shape)
    bins_masks = []
    for i, bin in enumerate(bins):
        if i < bins.shape[0] - 1:
            mask = np.logical_and(data >= bins[i], data <= bins[i + 1])
            binned_values[mask] = bin
            bins_masks.append(mask)
    return binned_values, bins, bins_masks


def compute_theta(pos_true, pos_pred, en_bin, pos_in_mm=True, folder='', net_name='', plot=True):
    if pos_in_mm:
        pos_true = pos_true * 0.00337  # in deg
        pos_pred = pos_pred * 0.00337  # in deg

    num_events = pos_pred.shape[0]
    theta_sq = np.sum((pos_true - pos_pred) ** 2, axis=1)

    hist_theta_sq, bins = np.histogram(theta_sq, bins=num_events)
    hist_theta_sq_normed = hist_theta_sq / float(num_events)
    cumsum_hist = np.cumsum(hist_theta_sq_normed)
    angular_resolution = np.sqrt(bins[np.where(cumsum_hist > 0.68)[0][0]])
    if not plot:
        return angular_resolution

    plt.figure()
    plt.hist(theta_sq, bins=80, log=True)
    plt.xlim([0, 0.4])
    plt.axvline(x=angular_resolution, color='darkorange', linestyle='--')
    plt.title(f'{net_name} Direction Reconstruction. Energy {en_bin}')
    plt.xlabel(r'$\theta^2$')
    plt.ylabel('Counts')
    plt.legend(['Angular Resolution: {:02e}'.format(angular_resolution)])
    plt.savefig(folder + '/' + net_name + '_angular_' + str(en_bin) + '.png')
    plt.savefig(folder + '/' + net_name + '_angular' + str(en_bin) + '.eps')

    return angular_resolution


def plot_angular_resolution(position_true, position_prediction, energy_true,
                            fig_folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction',
                            net_name=''):
    binned_values, bins, bins_masks = bin_data_mask(energy_true, 11)
    resolutions = []
    bin_medians = []

    for i, mask in enumerate(bins_masks):
        bin_pos = position_true[mask]
        bin_pred_pos = position_prediction[mask]
        bin_value = np.sqrt(bins[i] * bins[i + 1])
        res = compute_theta(bin_pos, bin_pred_pos, en_bin=bin_value, plot=True,
                            folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction/histograms')
        resolutions.append(res)
        bin_medians.append(bin_value)

    state_of_the_art_theta = np.array([0.157, 0.135, 0.108, 0.095, 0.081, 0.073, 0.071, 0.067, 0.065, 0.062, 0.056])
    state_of_the_art_energy = np.array([95, 150, 230, 378, 599, 949, 1504, 2383, 3777, 5986, 9487])

    plt.figure()
    plt.semilogx(10 ** np.array(bin_medians), resolutions, '-o')
    plt.semilogx(state_of_the_art_energy, state_of_the_art_theta, '--*')

    plt.xlabel('Energy')
    plt.ylabel('Angular Resolution')
    plt.title('Angular Resolution of ' + net_name)
    plt.legend([net_name, 'State of the art'])
    plt.grid()
    plt.savefig(fig_folder + '/angular_resolution' + net_name + '.png')
    plt.savefig(fig_folder + '/angular_resolution' + net_name + '.eps')
    plt.show()


def plot_angular_resolution_improvement(position_true_list, position_prediction_list, energy_true,
                        fig_folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction',
                        net_name='', makefigure=True):
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8 * 1.118, 8), sharex=True, gridspec_kw={'height_ratios': [1.618, 1]})
    # gs = gridspec.GridSpec(1, 2, height_ratios=[3, 1])
    marker_set = ['P', 'X', 'D', 'o', 's']
    for j, position_prediction in enumerate(position_prediction_list):

        binned_values, bins, bins_masks = bin_data_mask(energy_true[j], 11)
        resolutions = []
        bin_medians = []
        position_true = position_true_list[j]
        for i, mask in enumerate(bins_masks):
            bin_pos = position_true[mask]
            bin_pred_pos = position_prediction[mask]
            bin_value = np.sqrt(bins[i] * bins[i + 1])
            res = compute_theta(bin_pos, bin_pred_pos, en_bin=bin_value, plot=False,
                                folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction/histograms')
            resolutions.append(res)
            bin_medians.append(bin_value)

        ax1.semilogx(10 ** np.array(bin_medians[2:]), resolutions[2:], '--', marker=marker_set[j])
        ax1.grid(which='both', linestyle='--')

        state_of_the_art_theta = np.array([0.157, 0.135, 0.108, 0.095, 0.081, 0.073, 0.071, 0.067, 0.065, 0.062, 0.056])
        state_of_the_art_energy = np.array([95, 150, 230, 378, 599, 949, 1504, 2383, 3777, 5986, 9487])

        res_interp = np.interp(state_of_the_art_energy, 10 ** np.array(bin_medians), resolutions)
        enhancement = 100 * (state_of_the_art_theta - res_interp) / state_of_the_art_theta
        ax2.semilogx(np.array(state_of_the_art_energy), enhancement, '--', marker=marker_set[j])
        ax2.grid(which='both', linestyle='--')

    ax1.semilogx(state_of_the_art_energy, state_of_the_art_theta, '-*k', linewidth=3, markersize=10)
    ax2.semilogx([95, 9487], [0, 0], '-k', linewidth=3)
    # plt.xlim([100, 10000])
    # plt.ylim([0, 0.175])

    # ax1.set_xlabel('Energy (GeV)')
    ax1.set_ylabel('Angular Resolution')

    ax2.set_xlabel('Energy (GeV)')
    ax2.set_ylabel('Improvement (%)')

    ax1.set_title('68% Containment Angular Resolution of SE DenseNet-121', fontsize=18)
    ax1.legend(['SWA Taining II', 'SWA Training III', 'Minimum Validation', 'Minimum Validation No Time',
                'MAGIC Aleksic (2015)'])
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)

    # plt.grid(which='both', linestyle='--')
    plt.savefig(f'{fig_folder}/angular_resolution_TOTALE_4.png')
    plt.savefig(f'{fig_folder}/angular_resolution_TOTALE_4.pdf')

    # plt.show()
    plt.close()


