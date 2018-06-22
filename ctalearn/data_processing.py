import numpy as np
import cv2

from ctalearn.image_mapping import image_mapper

class DataProcessor():

    def __init__(self,
            image_mapper=image_mapper(None),
            crop=True,
            bounding_box_sizes={'MSTS': 48},
            image_cleaning="twolevel",
            thresholds={'MSTS': (5.5, 1.0)},
            return_cleaned_images=False,
            normalization="log",
            sort_images_by=None):
        
        self._image_mapper = image_mapper

        self.crop = crop
        self.bounding_box_sizes = bounding_box_sizes
        self.image_cleaning = image_cleaning
        self.thresholds = thresholds
        self.return_cleaned_images = return_cleaned_images

        self.normalization = normalization
    
        self.sort_images_by = sort_images_by

    # Crop an image about the shower center, optionally applying image cleaning
    # to obtain a better fit. The shower centroid is calculated as the mean of
    # pixel positions weighted by the charge, after cleaning. The cropped image is
    # obtained as a square bounding box centered on the centroid of side length
    # bounding_box_size.
    def _crop_image(self, image, tel_type):

        if self.image_cleaning == "none":
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
            m = (image_charge > self.picture_threshold).astype(np.uint8)
            # Dilate the mask once to add all adjacent pixels (i.e. kernel is 3x3)
            kernel = np.ones((3,3), np.uint8) 
            m = cv2.dilate(m, kernel)
            # Apply boundary threshold to keep weaker but adjacent pixels
            m = (m * image_charge > self.boundary_threshold).astype(np.uint8)
            m = np.expand_dims(m, 2)

            # Multiply by the mask to get the cleaned image
            cleaned_image = image * m
        else:
            raise ValueError('Unrecognized image cleaning method: {}'.format(
                image_cleaning_method))

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
        bounding_box_size = self.bounding_box_size[tel_type]
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
            image[:,:,0] = np.log(image[:,:,0] - self.dataset.image_charge_mins[tel_type] + 1.0)
        else:
            raise ValueError("Unrecognized normalization method {} selected.".format(self.normalization))

        return image

    # Function to apply all selected processing steps
    # on a given image. Returns the processed image and a
    # list of additional auxiliary parameters produced by
    # the processing.
    def _process_image(self, image, tel_type):
        auxiliary_info = []
        if self.crop:
            image, *shower_position = self._crop_image(image, tel_type)
            normalized_shower_position = [float(i)/self._image_mapper.IMAGE_SHAPES[tel_type][0] for i in shower_position] 
            auxiliary_info.append(normalized_shower_position)
        if self.normalization:
            image = self._normalize_image(image, tel_type)

        return image, auxiliary_info

    def process_example(self, data, label):
        
        # infer whether example is single tel or array based on
        # whether the dictionary values in data are numpy arrays
        # or lists of numpy arrays
        if isinstance(list(data.values())[0],list):
            example_type == "array"
        else:
            example_type == "single_tel"

        if example_type == "single_tel":
            tel_type = data.keys()[0]
            image = data[tel_type]

            image, _ = self._process_image(image, tel_type)
            data[tel_type] = image
            
            return [data, label]
        
        elif example_type == "array":
            for tel_type in data:
                images = data[tel_type][0]
                triggers = data[tel_type][1]
                aux_inputs = data[tel_type][2]
                if self.crop and tel_type in self.bounding_box_size:
                    image_shape = [self.bounding_box_size[tel_type], 
                        self._bounding_box_size[tel_type], 
                        self._image_mapper.IMAGE_SHAPES[tel_type][2]]
                else:
                    image_shape = self._image_mapper.IMAGE_SHAPES[tel_type]
                for i in range(len(images):
                    trigger = triggers[i]
                    if trigger == 0:
                        # telescope did not trigger, so provide a
                        # zeroed-out image
                        images[i] = np.zeros(image_shape))
                        if self.crop:
                            # add dummy centroid position to aux info
                            aux_info[i].extend([0, 0])
                    else:
                        image, auxiliary_info = self._process_image(images[i], tel_type)
                        images[i] = image
                        aux_inputs.extend(auxiliary_info)
          
                if self.sort_images_by == "trigger":
                    # Sort the images, triggers, and grouped auxiliary inputs by
                    # trigger, listing the triggered telescopes first
                    images, triggers, aux_inputs = map(list,
                            zip(*sorted(zip(images, triggers, aux_inputs), reverse=True, key=itemgetter(1))))
                elif self.sort_images_by == "size":
                    # Sort images by size (sum of charge in all pixels) from largest to smallest
                    images, triggers, aux_inputs = map(list,
                            zip(*sorted(zip(images, triggers, aux_inputs), reverse=True, key=lambda x: np.sum(x[0]))))
                
                # Convert to numpy arrays with correct types
                images = np.stack(images).astype(np.float32)
                triggers = np.array(telescope_triggers, dtype=np.int8)
                aux_inputs = np.array(aux_inputs, dtype=np.float32)

                data[tel_type] = [images, triggers, aux_inputs]

            return [data, label]

    # TODO: implement data augmentation.
    # Should be done at random each time an example is
    # processed, as the list of examples is fixed and determined
    # by DataLoader
    def _augment_data(self):
        raise NotImplementedError        
 


