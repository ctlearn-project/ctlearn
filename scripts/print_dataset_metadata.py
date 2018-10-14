import argparse
import sys

import tables
import numpy as np

from ctlearn.data_loading import HDF5DataLoader
from ctlearn.data_processing import DataProcessor

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Print metadata for a given collection of standard" 
            "CTA ML data files.")
        )
    parser.add_argument(
        'file_list',
        help="List of CTA ML HDF5 (pytables) files comprising a dataset."
        )
    parser.add_argument(
        '--out_file',
        help="Optional output file to write results to."
        )
    
    args = parser.parse_args()

    file_list = []
    with open(args.file_list) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                file_list.append(line)
	
    data_processing_settings = {}

    data_processor = DataProcessor(**data_processing_settings) 

    data_loader = HDF5DataLoader(file_list, 
            example_type="single_tel",
            data_processor=data_processor)

    metadata = data_loader.get_metadata()

    out = open(args.out_file,"w") if args.out_file else sys.stdout

    for i in metadata:
        print("{}: {}\n".format(i,metadata[i]), file=out)

    num_events_before_cuts_by_class_name = metadata['num_events_before_cuts_by_class_name']
    total_num_events = sum(num_events_before_cuts_by_class_name.values())
    print("{} total events.".format(total_num_events), file=out)
    print("Num events by particle_id:", file=out)
    for class_name in num_events_before_cuts_by_class_name:
        print("{}: {} ({}%)".format( 
                class_name, 
                num_events_before_cuts_by_class_name[class_name], 
                100 * float(num_events_before_cuts_by_class_name[class_name])/total_num_events),
                file=out
                )

    for tel_type in metadata['total_telescopes'].keys():
        print("\n" + tel_type + ":\n", file=out)
        num_images_before_cuts_by_class_name = metadata['num_images_before_cuts_by_tel_and_class_name'][tel_type]
        total_num_images = sum(num_images_before_cuts_by_class_name.values())
        print("{} total images.".format(total_num_images), file=out)
        print("Num images by particle_id:", file=out)
        for class_name in num_images_before_cuts_by_class_name:
            print("{}: {} ({}%)".format( 
                    class_name, 
                    num_images_before_cuts_by_class_name[class_name], 
                    100 * float(num_images_before_cuts_by_class_name[class_name])/total_num_images),
                    file=out
                    )

