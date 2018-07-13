import random
import os
import argparse

import numpy as np
import cv2
from PIL import Image, ImageDraw

from ctlearn.data_loading import HDF5DataLoader
from ctlearn.data_processing import DataProcessor
from ctlearn.image_mapping import ImageMapper

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Apply bounding boxes and segmentation on images and save for visualization."))
    parser.add_argument(
        'file_list',
        help='List of HDF5 (pytables) files to examine images from')
    parser.add_argument(
        "--num_images",
        help="number of randomly sampled images to examine",
        default=10)
    parser.add_argument(
        "--tel_type",
        help="type of telescope image to examine",
        default='MSTS')
    parser.add_argument(
        "--bounding_box_size",
        help="Side length of square bounding box",
        default=48)
    parser.add_argument(
        "--picture_threshold",
        help="First threshold (applied image-wide)",
        default=5.5)
    parser.add_argument(
        "--boundary_threshold",
        help="Second threshold (applied on pixels surviving the first cut)",
        default=1.0)
    parser.add_argument(
        "--save_original",
        help="whether to save the original (un-cleaned) image",
        action="store_true")
    parser.add_argument(
        "--save_cleaned",
        help="whether to save the cleaned image",
        action="store_true")
    parser.add_argument(
        "--save_cropped",
        help="wether to save the cropped image",
        action="store_true")
    parser.add_argument(
        "--save_dir",
        help="directory to save the image files",
        default='.')

    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)

    file_list = []
    with open(args.file_list) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                file_list.append(line)

    data_loader = HDF5DataLoader(file_list, 
            example_type="singletel", 
            selected_tel_type=args.tel_type,
            mode="test")

    data_processing_settings = {'crop': True,
                'bounding_box_sizes': {args.tel_type: args.bounding_box_size},
                'image_cleaning': 'twolevel',
                'thresholds': {args.tel_type: (args.picture_threshold, args.boundary_threshold)},
                'return_cleaned_images': False,
                'normalization': args.normalization,
                'image_charge_mins': data_loader.image_charge_mins 
                }

    data_processor = DataProcessor(**data_processing_settings) 

    count = 0
    while count < args.num_images:
        for run_number, event_number, tel_id in data_loader.examples:
        
            example = data_loader.get_example(run_number, event_number, tel_id)

            # get only the first channel (charge) of an image of arbitrary depth
            image_charge = example[0][:,:,0]

            if args.save_original:
                im = Image.fromarray(((image_charge - np.min(image_charge)) * 255 / np.max(image_charge)).astype('uint8'))
                im.save(os.path.join(save_dir,base_filename + '_{}_original.jpg'.format(i)))

            if args.save_cleaned:
                # get only the first channel (charge) of an image of arbitrary depth
                image_charge = example[0][:,:,0]

                data_processor.return_cleaned_images = True
                data_processor.bounding_box_sizes[args.tel_type] = ImageMapper.image_shapes[args.tel_type][0]

                cleaned_image, _, _ = data_processor._crop_image(image_charge, args.tel_type) 

                cleaned_image = Image.fromarray(((cleaned_image - np.min(cleaned_image)) * 255 / np.max(cleaned_image)).astype('uint8'))
                cleaned_image.save(os.path.join(save_dir,base_filename + '_{}_cleaned.jpg'.format(i)))

                data_processor.return_cleaned_images = False
                data_processor.bounding_box_sizes[args.tel_type] = args.bounding_box_size

            image_cropped, x_0, y_0 = data_processor._crop_image(image_charge, args.tel_type)

            if args.save_cropped:
                image_cropped = image_cropped[:,:,0]
                im_cropped = Image.fromarray(((image_cropped - np.min(image_cropped)) * 255 / np.max(image_cropped)).astype('uint8'))
                im_cropped.save(os.path.join(save_dir,base_filename + '_{}_cropped.jpg'.format(i)))

            x_min = int(round(x_0 - args.bounding_box_size/2))
            x_max = int(round(x_0 + args.bounding_box_size/2)) - 1
            y_min = int(round(y_0 - args.bounding_box_size/2))
            y_max = int(round(y_0 + args.bounding_box_size/2)) - 1

            im_box = Image.fromarray(((image_charge - np.min(image_charge)) * 255 / np.max(image_charge)).astype('uint8'))
            draw = ImageDraw.Draw(im_box)
            draw.rectangle([y_min, x_min, y_max, x_max],outline='yellow')
            im_box.save(os.path.join(save_dir,base_filename + '_{}_box.jpg'.format(i)))


