#!/usr/bin/python'

from image_mapping import *

import matplotlib.pyplot as plt
import numpy as np

plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)

test_mapper_over = ImageMapper(hex_conversion_algorithm='oversampling')
test_mapper_overPAD = ImageMapper(hex_conversion_algorithm='oversampling',padding = {
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
test_mapper_linear = ImageMapper(hex_conversion_algorithm='linear_interpolation')
test_mapper_linearPAD = ImageMapper(hex_conversion_algorithm='linear_interpolation',padding = {
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
test_mapper_cubic = ImageMapper(hex_conversion_algorithm='cubic_interpolation')
test_mapper_cubicPAD = ImageMapper(hex_conversion_algorithm='cubic_interpolation',padding = {
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
test_im_dict_over = {}
test_im_dict_overPAD = {}
test_im_dict_linear = {}
test_im_dict_linearPAD = {}
test_im_dict_cubic = {}
test_im_dict_cubicPAD = {}
test_pix_vals = {}

for tel_ in ['LST', 'MSTF', 'MSTN', 'MSTS', 'SST1', 'SSTC', 'SSTA', 'VTS', 'MGC', 'FACT','HESS-I','HESS-II']:
    
    test_pix_vals[tel_] = np.concatenate(([0.0],np.arange(0,test_mapper_over.num_pixels[tel_],1)),axis=0)
    test_pix_vals[tel_] = np.expand_dims(test_pix_vals[tel_], axis=1)
    
    # Oversampling
    test_im_dict_over[tel_] = test_mapper_over.map_image(test_pix_vals[tel_],tel_)
    test_im_dict_overPAD[tel_] = test_mapper_overPAD.map_image(test_pix_vals[tel_],tel_)
    
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
    plt.title('{} Camera - Oversampling'.format(tel_),fontsize=32)
    plt.axes().set_aspect('equal')
    plt.savefig('{}_Camera_Oversampling.png'.format(tel_), dpi = 300)
    plt.close()

    # Create vectorial grapics for each telescope camera with oversampling and padding.
    plt.figure(figsize=(20,20))
    plt.pcolor(test_im_dict_overPAD[tel_][:,:,0],cmap='viridis')
    '''
        Loop over data dimensions and create text annotations.
        In general, the text annotations refer to the pixel intensity. The intensity of
        the pixels are choosen that it's concurrently the index of the pixel of the
        telescope camera, expect for the MAGIC camera. "Outside" of the camera the
        intensity of the pixels are set to zero.
    '''
    if tel_ not in ['MGC']:
        for i in range(len(test_im_dict_overPAD[tel_][0])):
            for j in range(len(test_im_dict_overPAD[tel_][1])):
                if tel_ in ['SSTC','SSTA','MSTS']:
                    text_anno=int(test_im_dict_overPAD[tel_][i][j][0])
                else:
                    text_anno=int(test_im_dict_overPAD[tel_][i][j][0]*4)
                text = plt.text(j+0.5, i+0.5, text_anno,
                            ha="center", va="center", color="w",fontsize=2.5)
    plt.title('{} Camera - Oversampling (Padding)'.format(tel_),fontsize=32)
    plt.axes().set_aspect('equal')
    plt.savefig('{}_Camera_Oversampling_PAD.png'.format(tel_), dpi = 300)
    plt.close()

    if tel_ in ['LST', 'MSTF', 'MSTN', 'SST1', 'VTS', 'MGC', 'FACT','HESS-I','HESS-II']:
    
        # Linear interpolation
        test_im_dict_linear[tel_] = test_mapper_linear.map_image(test_pix_vals[tel_],tel_)
        test_im_dict_linearPAD[tel_] = test_mapper_linearPAD.map_image(test_pix_vals[tel_],tel_)
        # Create vectorial grapics for each telescope camera with linear interpolation.
        plt.figure(figsize=(20,20))
        plt.pcolor(test_im_dict_linear[tel_][:,:,0],cmap='viridis')
        
        plt.title('{} Camera - Linear interpolation'.format(tel_),fontsize=32)
        plt.axes().set_aspect('equal')
        plt.savefig('{}_Camera_LinearInterpolation.png'.format(tel_), dpi = 300)
        plt.close()

        # Create vectorial grapics for each telescope camera with linear interpolation and padding.
        plt.figure(figsize=(20,20))
        plt.pcolor(test_im_dict_linearPAD[tel_][:,:,0],cmap='viridis')
        
        plt.title('{} Camera - Linear interpolation (Padding)'.format(tel_),fontsize=32)
        plt.axes().set_aspect('equal')
        plt.savefig('{}_Camera_LinearInterpolation_PAD.png'.format(tel_), dpi = 300)
        plt.close()

        # Cubic interpolation
        test_im_dict_cubic[tel_] = test_mapper_cubic.map_image(test_pix_vals[tel_],tel_)
        test_im_dict_cubicPAD[tel_] = test_mapper_cubicPAD.map_image(test_pix_vals[tel_],tel_)
        # Create vectorial grapics for each telescope camera with cubic interpolation.
        plt.figure(figsize=(20,20))
        plt.pcolor(test_im_dict_cubic[tel_][:,:,0],cmap='viridis')
        
        plt.title('{} Camera - Cubic interpolation'.format(tel_),fontsize=32)
        plt.axes().set_aspect('equal')
        plt.savefig('{}_Camera_CubicInterpolation.png'.format(tel_), dpi = 300)
        plt.close()

        # Create vectorial grapics for each telescope camera with cubic interpolation and padding.
        plt.figure(figsize=(20,20))
        plt.pcolor(test_im_dict_cubicPAD[tel_][:,:,0],cmap='viridis')
        
        plt.title('{} Camera - Cubic interpolation (Padding)'.format(tel_),fontsize=32)
        plt.axes().set_aspect('equal')
        plt.savefig('{}_Camera_CubicInterpolation_PAD.png'.format(tel_), dpi = 300)
        plt.close()
