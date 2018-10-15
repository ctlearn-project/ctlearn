#!/usr/bin/python'

from image_mapping import *

import matplotlib.pyplot as plt
import numpy as np
    
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)

test_mapper_over = ImageMapper(hex_conversion_algorithm='oversampling')
test_mapper_linear = ImageMapper(hex_conversion_algorithm='linear_interpolation')
test_mapper_cubic = ImageMapper(hex_conversion_algorithm='cubic_interpolation')

test_im_dict_over = {}
test_im_dict_linear = {}
test_im_dict_cubic = {}
test_pix_vals = {}
    
for tel_ in ['LST', 'MSTF', 'MSTN', 'MSTS', 'SST1', 'SSTC', 'SSTA', 'VTS', 'MGC', 'FACT','HESS-I','HESS-II']:
    
    test_pix_vals[tel_] = np.concatenate(([0.0],np.arange(0,test_mapper_over.num_pixels[tel_],1)),axis=0)
    test_pix_vals[tel_] = np.expand_dims(test_pix_vals[tel_], axis=1)
    
    test_im_dict_over[tel_] = test_mapper_over.map_image(test_pix_vals[tel_],tel_)
    
    # Create vectorial grapics for each telescope camera with oversampling.
    plt.figure(figsize=(20,20))
    plt.pcolor(test_im_dict_over[tel_][:,:,0],cmap='viridis')
    '''
        Loop over data dimensions and create text annotations.
        In general, the text annotations refer to the pixel intensity. The intensity of
        the pixels are choosen that it's concurrently the index of the pixel of the
        telescope camera, expect for the MAGIC camera. "Outside" of the camera the
        intensity of the pixels are set to zero.
        '''
    if tel_ not in ['MGC']:
        for i in range(len(test_im_dict_over[tel_][0])):
            for j in range(len(test_im_dict_over[tel_][1])):
                if tel_ in ['SSTC','SSTA','MSTS']:
                    text_anno=int(test_im_dict_over[tel_][i][j][0])
                else:
                    text_anno=int(test_im_dict_over[tel_][i][j][0]*4)
                text = plt.text(j+0.5, i+0.5, text_anno,
                                ha="center", va="center", color="w",fontsize=2.5)
    plt.title('{} Camera Oversampling'.format(tel_),fontsize=42)
    plt.axes().set_aspect('equal')
    plt.savefig('{}_Camera_Oversampling.png'.format(tel_), dpi = 300)
    plt.close()

    if tel_ in ['LST', 'MSTF', 'MSTN', 'SST1', 'VTS', 'MGC', 'FACT','HESS-I','HESS-II']:
        
        test_im_dict_linear[tel_] = test_mapper_linear.map_image(test_pix_vals[tel_],tel_)
        # Create vectorial grapics for each telescope camera with linear interpolation.
        plt.figure(figsize=(20,20))
        plt.pcolor(test_im_dict_linear[tel_][:,:,0],cmap='viridis')
        
        plt.title('{} Camera Linear  Interpolation'.format(tel_),fontsize=42)
        plt.axes().set_aspect('equal')
        plt.savefig('{}_Camera_LinearInterpolation.png'.format(tel_), dpi = 300)
        plt.close()
        
        test_im_dict_cubic[tel_] = test_mapper_cubic.map_image(test_pix_vals[tel_],tel_)
        # Create vectorial grapics for each telescope camera with cubic interpolation.
        plt.figure(figsize=(20,20))
        plt.pcolor(test_im_dict_cubic[tel_][:,:,0],cmap='viridis')
        
        plt.title('{} Camera Cubic Interpolation'.format(tel_),fontsize=42)
        plt.axes().set_aspect('equal')
        plt.savefig('{}_Camera_CubicInterpolation.png'.format(tel_), dpi = 300)
        plt.close()
