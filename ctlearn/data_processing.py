from collections import namedtuple
from operator import itemgetter

import cv2
import numpy as np

from ctlearn.data_loading import get_camera_type
from ctlearn.image_mapping import ImageMapper

class DataProcessor():

    def __init__(self,
            image_mapper=ImageMapper(),
            crop=False,
            bounding_box_sizes=None,
            image_cleaning="twolevel",
            thresholds=None,
            return_cleaned_images=False,
            normalization=None,
            sorting=None
            ):
        
        self._image_mapper = image_mapper

        self.crop = crop

        if bounding_box_sizes is None:
            bounding_box_sizes = {'SCTCam': 48}
        self.bounding_box_sizes = bounding_box_sizes
        
        if image_cleaning in ['twolevel', None]:
            self.image_cleaning = image_cleaning
        else:
            raise ValueError("Invalid image cleaning method: {}. Select 'twolevel' or None.".format(image_cleaning))
        
        if thresholds is None:
            thresholds = {'SCTCam': (5.5, 1.0)}
        self.thresholds = thresholds
        self.return_cleaned_images = return_cleaned_images

        if normalization in ['log', None]:
            self.normalization = normalization
        else:
            raise ValueError("Invalid normalization method: {}. Select 'log' or None.".format(normalization))

        SortParams = namedtuple('SortParams', ['reverse', 'key'])
        self.sorting_params = {
                # List triggered telescopes first
                'trigger': SortParams(reverse=True, key=itemgetter(1)),
                # List from largest to smallest sum of pixel charges
                'size': SortParams(reverse=True, 
                    key=lambda x: np.sum(x[0]))
                }
        if sorting in list(self.sorting_params) + [None]:
            self.sorting = sorting
        else:
            raise ValueError("Invalid image sorting method: {}. Select "
                    "'trigger', 'size', or None.".format(sorting))

        self.image_shapes = {}
        for camera_type in self._image_mapper.image_shapes:
            if self.crop and camera_type in self.bounding_box_sizes:
                self.image_shapes[camera_type]= [self.bounding_box_sizes[camera_type],
                    self.bounding_box_sizes[camera_type],
                    self._image_mapper.image_shapes[camera_type][2]]
            else:
                self.image_shapes[camera_type] = self._image_mapper.image_shapes[camera_type]

        self.num_additional_aux_params = 0
        if self.crop:
            # Add 2 coordinates for the shower position on the camera
            self.num_additional_aux_params += 2

    # Crop an image about the shower center, optionally applying image cleaning
    # to obtain a better fit. The shower centroid is calculated as the mean of
    # pixel positions weighted by the charge, after cleaning. The cropped image is
    # obtained as a square bounding box centered on the centroid of side length
    # bounding_box_size.
    def _crop_image(self, image, camera_type):

        if self.image_cleaning is None:
            cleaned_image = image
        elif self.image_cleaning  == "twolevel":
            # Apply two-level cleaning to isolate the shower. First, filter for
            # shower pixels by applying a high charge cut (picture threshold).
            # Next, retain weaker pixels at the shower edge by allowing pixels
            # adjacent to those passing the first cut to pass a weaker cut
            # (boundary threshold).

            # Get only the first channel (charge) of an image of arbitrary depth
            image_charge = image[:,:,0]

            # Apply picture threshold to charge image to get mask
            m = (image_charge > self.thresholds[camera_type][0]).astype(np.uint8)
            # Dilate the mask once to add all adjacent pixels (i.e. kernel is 3x3)
            kernel = np.ones((3,3), np.uint8)
            m = cv2.dilate(m, kernel)
            # Apply boundary threshold to keep weaker but adjacent pixels
            m = (m * image_charge > self.thresholds[camera_type][1]).astype(np.uint8)
            m = np.expand_dims(m, 2)

            # Multiply by the mask to get the cleaned image
            cleaned_image = image * m

        # compute image moments, then use them to compute the centroid
        # coordinates (x_0, y_0)
        # NOTE: x_0 refers to a coordinate value along array axis 0 (rows, top to bottom)
        # y_0 refers to a coordinate value along array axis 1 (columns, left to right)
        # NOTE: when the image is blank after cleaning (sum of pixels is 0), set the
        # centroid to center of image to avoid divide by zero errors
        moments = cv2.moments(cleaned_image[:,:,0])
        x_0 = moments['m01']/moments['m00'] if moments['m00'] != 0 else image.shape[1]/2
        y_0 = moments['m10']/moments['m00'] if moments['m00'] != 0 else image.shape[0]/2

        # compute min and max x and y indices (along axis 0 and axis 1 respectively)
        # NOTE: these values are rounded and cast to integers, so they are valid indices
        # into the array
        # NOTE: rounding (and subtracting one from the max values) ensures that for all
        # float values of x_0, y_0, the values of indices x_min, x_max, y_min, y_max mark
        # a bounding box of exactly shape (BOUNDING_BOX_SIZE, BOUNDING_BOX_SIZE)
        bounding_box_size = self.bounding_box_sizes[camera_type]
        x_min = int(round(x_0 - bounding_box_size/2))
        x_max = int(round(x_0 + bounding_box_size/2)) - 1
        y_min = int(round(y_0 - bounding_box_size/2))
        y_max = int(round(y_0 + bounding_box_size/2)) - 1

        cropped_image = np.zeros((bounding_box_size,bounding_box_size,image.shape[2]),dtype=np.float32)

        # indices into the original image array which correspond to the bounding box region
        # when the bounding box goes over the edge of the original image array,
        # we truncate the appropriate indices so that all of x_min_image, x_max_image, etc.
        # are valid indices into the array
        x_min_image = x_min if x_min >= 0 else 0
        x_max_image = x_max if x_max <= (image.shape[0] - 1) else (image.shape[0] -1)
        y_min_image = y_min if y_min >= 0 else 0
        y_max_image = y_max if y_max <= (image.shape[1] - 1) else (image.shape[1] -1)

        # indices into the cropped image array of shape (BOUNDING_BOX_SIZE,BOUNDING_BOX_SIZE,image.shape[2])
        # which correspond to the region described by x_min, x_max, etc. in the original
        # image array. The region of the cropped image array which does not correspond to valid 
        # positions in the original image (the part which goes over the edges) are left filled
        # with zeros as padding.
        x_min_cropped = -x_min if x_min < 0 else 0
        x_max_cropped = (bounding_box_size - (x_max - x_max_image) - 1) if x_max >= (image.shape[0] - 1) else bounding_box_size - 1
        y_min_cropped = -y_min if y_min < 0 else 0
        y_max_cropped = (bounding_box_size - (y_max - y_max_image) - 1) if y_max >= (image.shape[1] - 1) else bounding_box_size - 1

        # transfer the cropped portion of the image array into the smaller, padded cropped_image array.
        # Use either the cleaned or uncleaned image as specified
        returned_image = (cleaned_image if self.return_cleaned_images else image)
        cropped_image[x_min_cropped:x_max_cropped+1,y_min_cropped:y_max_cropped+1,:] = returned_image[x_min_image:x_max_image+1,y_min_image:y_max_image+1,:]

        return cropped_image, x_0, y_0

    # Normalize the first channel of a given image
    # with the selected method
    def _normalize_image(self, image, tel_type):

        if self.normalization == "log":
            image[:,:,0] = np.log(image[:,:,0] - self.image_charge_mins[tel_type] + 1.0)
        else:
            raise ValueError("Unrecognized normalization method {} selected.".format(self.normalization))

        return image

    # Function to apply all selected processing steps
    # on a given image. Returns the processed image and a
    # list of additional auxiliary parameters produced by
    # the processing.
    def _process_image(self, image, tel_type, dummy_image=False):
        camera_type = get_camera_type(tel_type)
        auxiliary_input = []

        if dummy_image: # No trigger - image is blank
            if self.crop:
                # Add dummy centroid position to aux info
                auxiliary_input.extend([0.0, 0.0])
            return image, auxiliary_input

        if self.crop:
            image, *shower_position = self._crop_image(image, camera_type)
            image_width = self._image_mapper.image_shapes[camera_type][0]
            normalized_shower_position = [float(p) / image_width for p
                    in shower_position]
            auxiliary_input.extend(normalized_shower_position)
        if self.normalization:
            image = self._normalize_image(image, tel_type)

        return image, auxiliary_input

    # Process the example using the specified processing options
    def process_example(self, data, label, tel_types, example_type='array'):
        
        if example_type not in ['single_tel', 'array']:
            raise ValueError("Invalid example type selection: {}. Select "
                    "'single_tel' or 'array'.".format(example_type))
        
        # For single tel, just need to process the image
        if example_type == 'single_tel':
            image = data[0]
            image, _ = self._process_image(image, tel_types[0])
            data = [image]
            return data, label

        # Otherwise, array example. More complex processing is needed.
        for type_i, tel_type in enumerate(tel_types):
            for tel_i, tel_trigger in enumerate(data[type_i][1]):
                tel_image = data[type_i][0][tel_i]
                tel_aux_input = data[type_i][2][tel_i]
                dummy_image = True if tel_trigger == 0 else False
                tel_image, aux_input = self._process_image(tel_image, tel_type,
                        dummy_image=dummy_image)
                data[type_i][0][tel_i] = tel_image
                data[type_i][2][tel_i] = np.append(tel_aux_input,
                        aux_input).astype(np.float32)
      
            # Sort the images, triggers, and grouped auxiliary inputs together
            if self.sorting is not None:
                # Sort according to the specified parameters
                sort_params = self.sorting_params[self.sorting]
                data[type_i] = list(map(list, zip(*sorted(zip(*data[type_i]),
                        key=sort_params.key,
                        reverse=sort_params.reverse))))
        
        return data, label

    def get_metadata(self):
        
        metadata = {
                'image_shapes': self.image_shapes,
                'num_additional_aux_params': self.num_additional_aux_params
                }

        return metadata

    # TODO: implement data augmentation.
    # Should be done at random each time an example is
    # processed, as the list of examples is fixed and determined
    # by DataLoader
    def _augment_data(self):
        raise NotImplementedError        
 
    def add_image_charge_mins(self, image_charge_mins):
        self.image_charge_mins = image_charge_mins

