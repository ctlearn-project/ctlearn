import matplotlib.pyplot as plt
import numpy as np
import argparse
import datetime

if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Plots gpu utilization data.')
    parser.add_argument('log_file', help='path to log (CSV) file containing data')
    parser.add_argument('plot_name',help='plot name')
    args = parser.parse_args()

    data = np.genfromtxt(args.log_file, delimiter=',', skip_header=1, names=['timestamp', 'gpu_util', 'gpu_mem_util'])

    x = range(len(data['gpu_util']))
    avg_gpu_util = np.mean(data['gpu_util'])
    avg_mem_util = np.mean(data['gpu_mem_util'])

    plt.plot(x,data['gpu_util'],color='blue',label='gpu utilization')
    plt.plot(x,data['gpu_mem_util'],color='green',label='gpu memory utilization')

    plt.axhline(y=avg_gpu_util,color='blue',linestyle='--',label='gpu utilization (mean)')
    plt.axhline(y=avg_mem_util,color='green',linestyle='--',label='gpu memory utilization (mean)')

    plt.legend()
    plt.title('GPU Utilization')
    plt.xlabel('Sample (arbitrary units)')
    plt.ylabel('Percent Utilization')

    print('Average GPU Utilization: {}'.format(avg_gpu_util))
    print('Average GPU Memory Utilization: {}'.format(avg_mem_util))

    plt.savefig(args.plot_name)


