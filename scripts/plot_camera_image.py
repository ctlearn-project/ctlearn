#!/usr/bin/python'

from image_mapping import *

import matplotlib.pyplot as plt
import numpy as np
    
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)

test_mapper = ImageMapper()
test_mapper_pad = ImageMapper(padding = {
                                      'LST': 10,
                                      'MSTF': 10,
                                      'MSTN': 20,
                                      'MSTS': 10,
                                      'SST1': 10,
                                      'SSTC': 10,
                                      'SSTA': 10,
                                      'VTS': 10,
                                      'MGC': 5,
                                      'FACT': 10,
                                      'HESS-I': 10,
                                      'HESS-II': 10
                                      })
    
test_im_dict = {}
test_im_dict_pad = {}
test_pix_vals = {}
    
for tel_ in ['LST', 'MSTF', 'MSTN', 'MSTS', 'SST1', 'SSTC', 'SSTA', 'VTS', 'MGC', 'FACT','HESS-I','HESS-II']:

    test_pix_vals[tel_] = np.concatenate(([0.0],np.arange(0,test_mapper.num_pixels[tel_],1)),axis=0)
    test_pix_vals[tel_] = np.expand_dims(test_pix_vals[tel_], axis=1)
    test_im_dict[tel_] = test_mapper.map_image(test_pix_vals[tel_],tel_)
    test_im_dict_pad[tel_] = test_mapper_pad.map_image(test_pix_vals[tel_],tel_)
        
    # Create vectorial grapics for each telescope camera.
    plt.figure(figsize=(20,20))
    plt.pcolor(test_im_dict[tel_][:,:,0],cmap='viridis')
    '''
        Loop over data dimensions and create text annotations.
        In general, the text annotations refer to the pixel intensity. The intensity of
        the pixels are choosen that it's concurrently the index of the pixel of the
        telescope camera, expect for the MAGIC camera. "Outside" of the camera the
        intensity of the pixels are set to zero.
    '''
    for i in range(len(test_im_dict[tel_][0])):
        for j in range(len(test_im_dict[tel_][1])):
            if tel_ in ['SSTC','SSTA','MSTS']:
                text_anno=int(test_im_dict[tel_][i][j][0])
            else:
                text_anno=int(test_im_dict[tel_][i][j][0]*4)
            text = plt.text(j+0.5, i+0.5, text_anno,
                                ha="center", va="center", color="w",fontsize=2.5)
    plt.title('{} Camera'.format(tel_),fontsize=42)
    plt.axes().set_aspect('equal')
    plt.savefig('{}_Camera.png'.format(tel_), dpi = 300)
    plt.close()
        
    # Create vectorial grapics for each telescope camera with padding.
    plt.figure(figsize=(20,20))
    plt.pcolor(test_im_dict_pad[tel_][:,:,0],cmap='viridis')
    '''
        Loop over data dimensions and create text annotations.
        In general, the text annotations refer to the pixel intensity. The intensity of
        the pixels are choosen that it's concurrently the index of the pixel of the
        telescope camera, expect for the MAGIC camera. "Outside" of the camera the
        intensity of the pixels are set to zero.
    '''
    for i in range(len(test_im_dict_pad[tel_][0])):
        for j in range(len(test_im_dict_pad[tel_][1])):
            if tel_ in ['SSTC','SSTA','MSTS']:
                text_anno=int(test_im_dict_pad[tel_][i][j][0])
            else:
                text_anno=int(test_im_dict_pad[tel_][i][j][0]*4)
            text = plt.text(j+0.5, i+0.5, text_anno,
                            ha="center", va="center", color="w",fontsize=2.5)
    plt.title('{} Camera (Padding)'.format(tel_),fontsize=42)
    plt.axes().set_aspect('equal')
    plt.savefig('{}_Camera_Padding.png'.format(tel_), dpi = 300)
    plt.close()
