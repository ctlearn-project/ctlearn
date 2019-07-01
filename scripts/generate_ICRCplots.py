import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import argparse

def generate_plot(path,tel_type,curve_type):
    
    files = np.array([x for x in os.listdir(path) if x.endswith(".csv")])
    
    if tel_type is None:
        tel_type = ['LSTCam','FlashCam','DigiCam']
    else:
        if tel_type not in ['LSTCam','FlashCam','DigiCam']:
            raise ValueError("Invalid tel_type: {}".format(tel_type))
        tel_type = [tel_type]
    if curve_type is None:
        curve_type = ['accuracy','auc','loss']
    else:
        if curve_type not in ['accuracy','auc','loss']:
            raise ValueError("Invalid curve_type: {}".format(curve_type))
        curve_type = [curve_type]

    mapping_methods = ['oversampling','rebinning','nearest_interpolation','bilinear_interpolation','bicubic_interpolation']

    for tel in tel_type:
        for curve in curve_type:
            counter = 0
            for map_method in mapping_methods:
                for file in files:
                    if tel in file and curve in file and map_method in file:
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
                        if counter is 0:
                            y_ref = y_val
                            fig = plt.figure()
                            plt.title("{} - {}".format(tel,curve))
                            plt.xticks([])
                            plt.yticks([])
                        frame1 = fig.add_axes((0.1,0.3,0.8,0.6))
                        plt.plot(x_val,y_val,label=map_method)
                        counter+=1
                        if counter is 5:
                            if curve is 'loss':
                                plt.legend(loc='upper right', fontsize=15, fancybox=True, framealpha=0.2, labelspacing=0.05)
                            else:
                                plt.legend(loc='lower right', fontsize=15, fancybox=True, framealpha=0.2, labelspacing=0.05)
                        plt.grid(b=True,color='grey', linestyle='--', linewidth=0.25)
                        frame1.set_xticklabels([])
                        #frame1.set_yticklabels([])
                        frame2=fig.add_axes((0.1,0.1,0.8,0.2))
                        plt.plot(x_val,np.subtract(y_val,y_ref))

                        if counter is 5:
                            plt.savefig("{}{}_{}.png".format(path,tel,curve))
                            print("{}{}_{}.png".format(path,tel,curve))
                            plt.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description=("Generate plots for the ICRC."))
    parser.add_argument(
            'path_to_csvfile',
            help="path to the tensorflow output (csv file)")
    parser.add_argument(
            '--tel_type',
            default=None,
            help="Select a specific telescope type. Default: all (LSTCam, FlashCam, DigiCam).")
    parser.add_argument(
            '--curve_type',
            default=None,
            help="Select a specific curve type. Default: all (accuracy, auc, loss).")
            
    args = parser.parse_args()

    generate_plot(path=args.path_to_csvfile,tel_type=args.tel_type,curve_type=args.curve_type)
