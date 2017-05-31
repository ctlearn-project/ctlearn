import argparse
import os
from shutil import copyfile, copy2
import sys
import random

# Class directory names
gamma_data_dir = "gamma-diffuse"
proton_data_dir = "proton"

# parse command line arguments

parser = argparse.ArgumentParser(description='Prep dataset')
parser.add_argument('source_train_data_dir', help = 'path to source train data directory')
parser.add_argument('source_val_data_dir', help = 'path to source val data directory')
parser.add_argument('train_data_dir', help='path to final training data directory (containing subdir for each type)')
parser.add_argument('val_data_dir', help='path to final validation data directory (containing subdir for each type)')
parser.add_argument('train_samples', type=int,help='train samples (50-50)')
parser.add_argument('val_samples', type=int,help='validation samples (50-50)')

args = parser.parse_args()

def copy_link(src, dst):
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    else:
        shutil.copy(src,dst)

for directory in [os.path.join(args.train_data_dir,gamma_data_dir),os.path.join(args.train_data_dir,proton_data_dir),os.path.join(args.val_data_dir,gamma_data_dir),os.path.join(args.val_data_dir,proton_data_dir)]:

    if not os.path.exists(directory):
        os.makedirs(directory)

for directory in [args.source_train_data_dir,args.source_val_data_dir]:

    dict_gamma = {}
    dict_proton = {}

    directory_path = os.path.abspath(directory)

    if directory == args.source_train_data_dir:
        samples = args.train_samples
        target_dir = args.train_data_dir
    else:
        samples = args.val_samples
        target_dir = args.val_data_dir

    for ptype in ['gamma','proton']:
        if ptype == 'gamma':
            data_dir = os.path.join(directory_path,gamma_data_dir)
            dir_name = gamma_data_dir
            event_dict = dict_gamma
        else:
            data_dir = os.path.join(directory_path,proton_data_dir)
            dir_name = proton_data_dir
            event_dict = dict_proton

        for filename in os.listdir(data_dir):
            if filename.endswith(".png"):
                #get event ID and energy
                event_id, energy, impact, tel_num = filename.rsplit(".",1)[0].split("_")

                if event_id in event_dict:
                    event_dict[event_id][tel_num] = filename
                else:
                    event_dict[event_id] = {tel_num:filename}

        #unique event ids
        event_count = len(event_dict)
        print("Total {} events: {}".format(ptype,event_count))

        keys = list(event_dict.keys())
        random.shuffle(keys)

        keys = keys[0:samples/2]

        for event_id in keys:
            tel_dict = event_dict[event_id]
            for tel_num in tel_dict:
                print(os.path.join(data_dir,tel_dict[tel_num]))
                print(os.path.join(target_dir,dir_name,tel_dict[tel_num]))
                copy_link(os.path.join(data_dir,tel_dict[tel_num]),os.path.join(target_dir,dir_name,tel_dict[tel_num]))


