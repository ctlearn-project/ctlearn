#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:24:37 2018

@author: jsevillamol
"""

import argparse
import sys
import yaml

from ctlearn.data_loading import HDF5DataLoader
from ctlearn.image_mapping import ImageMapper
from ctlearn.data_processing import DataProcessor

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Print train metadata for a .yaml run configuration")
        )
    parser.add_argument(
        'config_file',
        help=".yaml file with the run configuration"
        )
    parser.add_argument(
        '--out_file',
        help="Optional output file to write results to."
        )
    
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as config_file:
        config = yaml.load(config_file)
    

    # Load options related to the data format and location
    data_format = config['Data']['format']
    data_files = []
    with open(config['Data']['file_list']) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                data_files.append(line)
	
    data_loading_settings = config['Data'].get('Loading', {})

    # Load options related to data processing
    apply_processing = config['Data'].get('apply_processing', True)
    data_processing_settings = config['Data'].get('Processing', {})
    
    # Load options related to image mapping
    image_mapping_settings = config.get('Image Mapping', {})
    if 'use_peak_times' in config['Data']['Loading']:
        image_mapping_settings['use_peak_times'] = config['Data']['Loading']['use_peak_times']
    else:
        image_mapping_settings['use_peak_times'] = False

    # Load options related to data loading
    data_loader_mode = "train"
    
    # Create data processor
    if apply_processing:
        data_processor = DataProcessor(
                image_mapper=ImageMapper(**image_mapping_settings),
                **data_processing_settings)
    else: data_processor = None

    # Define data loading functions
    if data_format == 'HDF5':
        data_loader = HDF5DataLoader(
                data_files,
                mode=data_loader_mode,
                data_processor=data_processor,
                image_mapper=ImageMapper(**image_mapping_settings),
                **data_loading_settings)

    metadata = data_loader.get_metadata()

    out = open(args.out_file,"w") if args.out_file else sys.stdout
    
    print("Tel {} ({} mode)\n".format(config['Data']['Loading']['selected_tel_type'], config['Data']['Loading']['example_type']), file=out)
    
    for i in metadata:
        print("{}: {}\n".format(i,metadata[i]), file=out)

    num_events_after_cuts_by_class_name = metadata['num_events_after_cuts_by_class_name']
    total_num_events = sum(num_events_after_cuts_by_class_name.values())
    print("\n{} total events.".format(total_num_events), file=out)
    print("Num events by particle_id:", file=out)
    for class_name in num_events_after_cuts_by_class_name:
        print("{}: {} ({}%)".format( 
                class_name, 
                num_events_after_cuts_by_class_name[class_name], 
                100 * float(num_events_after_cuts_by_class_name[class_name])/total_num_events),
                file=out
                )

    num_images_after_cuts_by_class_name = metadata['num_images_after_cuts_by_class_name']
    total_num_images = sum(num_images_after_cuts_by_class_name.values())
    print("\n{} total images.".format(total_num_images), file=out)
    print("Num images by particle_id:", file=out)
    for class_name in num_images_after_cuts_by_class_name:
        print("{}: {} ({}%)".format( 
                class_name, 
                num_images_after_cuts_by_class_name[class_name], 
                100 * float(num_images_after_cuts_by_class_name[class_name])/total_num_images),
                file=out
                )
            
    num_val_examples_by_class_name = metadata['num_val_examples_by_class_name']
    total_num_events = sum(num_val_examples_by_class_name.values())
    print("\n{} total val examples.".format(total_num_events), file=out)
    print("Num val examples by particle_id:", file=out)
    for class_name in num_val_examples_by_class_name:
        print("{}: {} ({}%)".format( 
                class_name, 
                num_val_examples_by_class_name[class_name], 
                100 * float(num_val_examples_by_class_name[class_name])/total_num_events),
                file=out
                )
    