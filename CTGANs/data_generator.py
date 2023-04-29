import numpy as np
import matplotlib.pyplot as plt


def plot_grid(images, nrows=5, ncols=5, save_path=''):
  channels = images.shape[-1]
  # Create a figure
  fig = plt.figure(constrained_layout=True, figsize=(ncols*channels*2, nrows*2))
  # Create one subfigure for every image, that is for every position in the grid
  subfigs = fig.subfigures(nrows=nrows, ncols=ncols, squeeze=False, wspace=0.1, hspace=0.1)
  for i in range(nrows):
    for j in range(ncols):
      # Create, for every subfigure (that is for every image or grid position), one axes for every image channel
      axes = subfigs[i, j].subplots(nrows=1, ncols=channels, squeeze=False, gridspec_kw={'wspace': 0})
      axes = axes.flatten()
      for k in range(channels):
        # Plot the k channel for the image in the (i, j) grid position
        axes[k].pcolor(np.squeeze(images[i*ncols+j, :, :, k]), cmap='viridis')
        axes[k].axis('off')
        # Write the title of the channels only for the first row
        if i == 0 and channels > 1:
          axes[k].set_title(f'Channel {k+1}')
  
  # Save the plot if a path is specified
  if save_path:
    fig.savefig(save_path)