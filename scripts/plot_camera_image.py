from ctlearn.image_mapping import *

import matplotlib.pyplot as plt
import numpy as np

plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)

hex_algo = ['oversampling','nearest_interpolation','rebinning','bilinear_interpolation','bicubic_interpolation']

for i in hex_algo:
    print('Hex_conversion_algorithm: {}'.format(i))
    test_mapper = ImageMapper(camera_type=['LSTCam','ASTRICam', 'SCTCam', 'CHEC', 'MAGICCam','FlashCam', 'NectarCam', 'VERITAS', 'FACT', 'HESS-I', 'HESS-II'], hex_conversion_algorithm={'LSTCam':i,'ASTRICam':i, 'SCTCam':i, 'CHEC':i, 'MAGICCam':i,'FlashCam':i, 'NectarCam':i, 'VERITAS':i, 'FACT':i, 'HESS-I':i, 'HESS-II':i},interpolation_image_shape={'LSTCam':(120, 120, 1),'ASTRICam':(120, 120, 1), 'SCTCam':(125, 125, 1), 'CHEC':(120, 120, 1), 'MAGICCam':(120, 120, 1),'FlashCam':(120, 120, 1), 'NectarCam':(120, 120, 1), 'VERITAS':(120, 120, 1), 'FACT':(120, 120, 1), 'HESS-I':(120, 120, 1), 'HESS-II':(120, 120, 1)})
    
    test_mapper_PAD = ImageMapper(camera_type=['LSTCam','ASTRICam', 'SCTCam', 'CHEC', 'MAGICCam','FlashCam', 'NectarCam', 'VERITAS', 'FACT', 'HESS-I', 'HESS-II'], hex_conversion_algorithm={'LSTCam':i,'ASTRICam':i, 'SCTCam':i, 'CHEC':i, 'MAGICCam':i,'FlashCam':i, 'NectarCam':i, 'VERITAS':i, 'FACT':i, 'HESS-I':i, 'HESS-II':i},interpolation_image_shape={'LSTCam':(120, 120, 1),'ASTRICam':(120, 120, 1), 'SCTCam':(125, 125, 1), 'CHEC':(120, 120, 1), 'MAGICCam':(120, 120, 1),'FlashCam':(120, 120, 1), 'NectarCam':(120, 120, 1), 'VERITAS':(120, 120, 1), 'FACT':(120, 120, 1), 'HESS-I':(120, 120, 1), 'HESS-II':(120, 120, 1)},padding = {'LSTCam':10,'ASTRICam':10, 'SCTCam':10, 'CHEC':10, 'MAGICCam':10,'FlashCam':10, 'NectarCam':10, 'VERITAS':10, 'FACT':10, 'HESS-I':10, 'HESS-II':10})
    
    test_im_dict = {}
    test_im_dict_PAD = {}
    
    test_pix_vals = {}
    
    for tel_ in ['LSTCam','ASTRICam', 'SCTCam', 'CHEC', 'MAGICCam','FlashCam', 'NectarCam', 'VERITAS', 'FACT', 'HESS-I', 'HESS-II']:
    
        test_pix_vals[tel_] = np.concatenate(([0.0],np.arange(0,test_mapper.num_pixels[tel_],1)),axis=0)
        test_pix_vals[tel_] = np.expand_dims(test_pix_vals[tel_], axis=1)
    
        test_im_dict[tel_] = test_mapper.map_image(test_pix_vals[tel_],tel_)
        test_im_dict_PAD[tel_] = test_mapper_PAD.map_image(test_pix_vals[tel_],tel_)
    
        # Create vectorial grapics for each telescope camera with oversampling.
        plt.figure(figsize=(20,20))
        plt.pcolor(test_im_dict[tel_][:,:,0],cmap='viridis')
        plt.title('{} - {}'.format(tel_,i),fontsize=52)
        plt.axes().set_aspect('equal')
        plt.savefig('{}_{}.png'.format(tel_,i), dpi = 300)
        plt.close()
        
        # Create vectorial grapics for each telescope camera with oversampling and padding.
        plt.figure(figsize=(20,20))
        plt.pcolor(test_im_dict_PAD[tel_][:,:,0],cmap='viridis')
        plt.title('{} - {} (Padding)'.format(tel_,i),fontsize=52)
        plt.axes().set_aspect('equal')
        plt.savefig('{}_{}PAD.png'.format(tel_,i), dpi = 300)
        plt.close()
