#
#
#  Run this script with the path of the folder containing the csv files.
#  $ python generate_ICRC_CTLearnplots.py ./tensorborad_output_csv/
#  (No further arguments means all telescope types will be included
#   in each curve plot.)
#
#

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import argparse

def generate_plot(path,tel_type,curve_type):
    
    files = np.array([x for x in os.listdir(path) if x.endswith(".csv")])
    
    if tel_type is None:
        tel_type = ['LSTCam','FlashCam','DigiCam','NectarCam','CHEC','ASTRICam','SCTCam']
    else:
        if tel_type not in ['LSTCam','FlashCam','DigiCam','NectarCam','CHEC','ASTRICam','SCTCam']:
            raise ValueError("Invalid tel_type: {}".format(tel_type))
        tel_type = [tel_type]
    if curve_type is None:
        curve_type = ['accuracy','auc','loss']
    else:
        if curve_type not in ['accuracy','auc','loss']:
            raise ValueError("Invalid curve_type: {}".format(curve_type))
        curve_type = [curve_type]

    for curve in curve_type:
        fig = plt.figure(figsize=(8,8))
        counter=0
        for tel in tel_type:
            print(tel)
            for file in files:
                if tel in file and curve in file:
                    print(path+file)
                    data = open(path+file)
                    dummy_data = csv.reader(data, delimiter=',')
                    x_val = []
                    y_val = []
                    for row in dummy_data:
                        x_val.append(row[1])
                        y_val.append(row[2])
                    x_val = np.array(x_val[1:]).astype(np.float)
                    y_val = np.array(y_val[1:]).astype(np.float)
                    plt.plot(x_val,y_val,linestyle='-',c=map_color[tel],label=tel)
        if curve is 'accuracy':
            plt.legend(loc='lower right', fontsize=28, fancybox=True, framealpha=0.7, labelspacing=0.05)
        plt.tick_params(labelsize=23)
        plt.grid(b=True,color='grey', linestyle='--', linewidth=0.25)
        plt.xlim(np.min(x_val),np.max(x_val))
        plt.savefig("{}ICRC_CTLearn_{}_cnnrnn_nolabels.pdf".format(path,curve),format='pdf',dpi=300)
        print("{}ICRC_CTLearn_{}_cnnrnn_nolabels.pdf".format(path,curve))
        plt.close()

if __name__ == "__main__":
                              
    map_color = {'LSTCam':'b', 'FlashCam':'r', 'DigiCam':'c', 'NectarCam':'orange', 'CHEC':'indigo','ASTRICam':'g','SCTCam':'olive'}
    
    parser = argparse.ArgumentParser(
            description=("Generate plots for the ICRC."))
    parser.add_argument(
            'path_to_csvfile',
            help="path to the tensorflow output (csv file)")
    parser.add_argument(
            '--tel_type',
            default=None,
            help="Select a specific telescope type. Default: all (LSTCam, FlashCam, DigiCam, NectarCam, CHEC, ASTRICam, SCTCam).")
    parser.add_argument(
            '--curve_type',
            default=None,
            help="Select a specific curve type. Default: all (accuracy, auc, loss).")
            
    args = parser.parse_args()

    generate_plot(path=args.path_to_csvfile,tel_type=args.tel_type,curve_type=args.curve_type)
