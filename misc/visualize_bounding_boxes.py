import random
import os
import argparse

import numpy as np
import cv2
from PIL import Image, ImageDraw

from ctalearn.data import crop_image, load_image_HDF5, load_metadata_HDF5, return_file_handle

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Apply bounding boxes and segmentation on images and save for visualization."))
    parser.add_argument(
        'file_list',
        help='List of HDF5 (pytables) files to examine images from')
    parser.add_argument(
        "--images_per_file",
        help="number of randomly sampled images to examine per file",
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

    metadata = load_metadata_HDF5(file_list)

    settings = {'image_cleaning_method': 'twolevel',
                'return_cleaned_images': False,
              'bounding_box_size': args.bounding_box_size, 
              'picture_threshold': args.picture_threshold, 
              'boundary_threshold': args.boundary_threshold}

    index_lists = [random.sample(range(1, num_images_total), args.images_per_file) for num_images_total in metadata['num_images_by_file'][args.tel_type]]

    for filename, indices in zip(file_list, index_lists):
        for i in indices:

            f_data = return_file_handle(filename.encode('utf-8'))
            image = load_image_HDF5(f_data, args.tel_type ,i)

            # get only the first channel (charge) of an image of arbitrary depth
            image_charge = image[:,:,0]

            base_filename = os.path.basename(filename)
            base_filename , _ = os.path.splitext(base_filename)

            if args.save_original:
                im = Image.fromarray(((image_charge - np.min(image_charge)) * 255 / np.max(image_charge)).astype('uint8'))
                im.save(os.path.join(save_dir,base_filename + '_{}_original.jpg'.format(i)))

            if args.save_cleaned:
                # get only the first channel (charge) of an image of arbitrary depth
                image_charge = image[:,:,0]

                # apply picture threshold to charge image to get mask, then dilate the mask once 
                # to add all adjacent pixels (i.e. kernel is 3x3)
                m = (image_charge > args.picture_threshold).astype(np.uint8)
                kernel = np.ones((3,3), np.uint8) 
                m = cv2.dilate(m,kernel)

                # multiply the charge image by the dilated mask
                # then mask out all surviving pixels which are smaller than
                # the boundary threshold
                image_cleaned = m * image_charge 
                image_cleaned = image_cleaned * (image_cleaned > args.boundary_threshold).astype(np.uint8)

                im_cleaned = Image.fromarray(((image_cleaned - np.min(image_cleaned)) * 255 / np.max(image_cleaned)).astype('uint8'))
                im_cleaned.save(os.path.join(save_dir,base_filename + '_{}_cleaned.jpg'.format(i)))

            image_cropped, x_0, y_0 = crop_image(image, settings)

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


