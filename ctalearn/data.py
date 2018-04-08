from operator import itemgetter
import threading
import logging
import math
from collections import OrderedDict
import random

import tables
import numpy as np
import cv2

from ctalearn.image import MAPPING_TABLES, IMAGE_SHAPES

logger = logging.getLogger(__name__)

# dict mapping CORSIKA particle ids to class number
PARTICLE_ID_TO_CLASS = {101:0, 0:1}
# dict mapping class number to particle name
CLASS_TO_NAME = {0:'proton',1:'gamma'}

# Multithread-safe PyTables open and close file functions
# See http://www.pytables.org/latest/cookbook/threading.html
lock = threading.Lock()

def synchronized_open_file(*args, **kwargs):
    with lock:
        return tables.open_file(*args, **kwargs)

def synchronized_close_file(self, *args, **kwargs):
    with lock:
        return self.close(*args, **kwargs)

# Externally store the file handles corresponding to each filename.
# This structures allow the load_data functions to read from HDF5 files without
# the expensive need to open and close them for each event.
# NOTE: this function makes use of the fact that dicts as default arguments are
# mutable. That is, after something is added to file_handle_dict in one
# function call, it will still be there the next time the function is called.
def return_file_handle(filename, file_handle_dict={}):
    if filename not in file_handle_dict:
        file_handle_dict[filename] = synchronized_open_file(
                filename.decode('utf-8'), mode='r')
    return file_handle_dict[filename]

# Data loading function for event-wise (array-level) HDF5 data loading
def load_data_eventwise_HDF5(filename, index, auxiliary_data, metadata,
        settings):

    # Read the event record for the given filename and index
    f = return_file_handle(filename)
    record = f.root.Event_Info[index]
    
    # Get classification label by converting CORSIKA particle code
    gamma_hadron_label = PARTICLE_ID_TO_CLASS[record['particle_id']]
   
    # Collect image indices (indices into the image tables)
    # for each telescope type in this event
    telescope_types = settings['processed_telescope_types']
    image_indices = {tel_type:record[tel_type+"_indices"] for tel_type in
            telescope_types}
    # Collect images, auxiliary info, and binary trigger values
    telescope_images = []
    telescope_triggers = []
    shower_positions = [] # only used when cropping images
    for tel_type in telescope_types:
        image_shape = settings['processed_image_shapes'][tel_type]
        for i in image_indices[tel_type]:
            if i == 0:
                # Telescope did not trigger. Its outputs will be dropped
                # out, so input is arbitrary. Use an empty array for
                # efficiency.
                telescope_images.append(np.zeros(image_shape))
                if settings['crop_images']:
                    shower_positions.append([0, 0])
                telescope_triggers.append(0)
            else:
                telescope_image = load_image_HDF5(f, tel_type, i)
                if settings['crop_images']:
                    telescope_image, *shower_position = crop_image(
                            telescope_image, settings)
                    shower_positions.append([float(i)/metadata['image_shapes'][tel_type][0] for i in shower_position])
                if settings['log_normalize_charge']:
                    telescope_image[:,:,0] = np.log(telescope_image[:,:,0] - metadata['image_charge_min'][tel_type] + 1.0)
                telescope_images.append(telescope_image)
                telescope_triggers.append(1)
   
    if settings['use_telescope_positions']:
        telescope_positions = []
        for tel_type in telescope_types:
            # Collect telescope positions from auxiliary data
            # telescope_positions is a list of lists
            # ex. [[x1,y1,z1],[x2,y2,z2],...]
            for tel_id in sorted(auxiliary_data['telescope_positions'][tel_type].keys()):
                # normalize the x, y and z coordinates in the telescope position based on the maximum value of each
                x, y, z = auxiliary_data['telescope_positions'][tel_type][tel_id]
                tel_pos = [float(x)/metadata['max_telescope_pos'][0], float(y)/metadata['max_telescope_pos'][1], float(z)/metadata['max_telescope_pos'][2]] 
                telescope_positions.append(tel_pos)

    # Construct telescope auxiliary inputs as specified
    telescope_aux_inputs = []
    for aux_input in settings['processed_aux_input_nums'].keys():
        if aux_input == 'telescope_position':
            telescope_aux_inputs.append(telescope_positions)
        elif aux_input == 'shower_position':
            telescope_aux_inputs.append(shower_positions)
    # Group parameters by telescope
    telescope_aux_inputs = [tel_params for [*tel_params] in
            zip(*telescope_aux_inputs)]
    # For each telescope, merge the parameters into a single list
    telescope_aux_inputs = [[param for param_list in tel_list for param in
        param_list] for tel_list in telescope_aux_inputs]

    if settings['sort_telescopes_by_trigger']:
        # Sort the images, triggers, and grouped auxiliary inputs by
        # trigger, listing the triggered telescopes first
        """
        telescope_images, telescope_triggers, telescope_aux_inputs = map(list,
                zip(*sorted(zip(telescope_images, telescope_triggers,
                    telescope_aux_inputs), reverse=True, key=itemgetter(1))))
        """

        telescope_images, telescope_triggers, telescope_aux_inputs = map(list,
                zip(*sorted(zip(telescope_images, telescope_triggers,
                    telescope_aux_inputs), reverse=True, key=lambda x: np.sum(x[0]))))
      
    # Convert to numpy arrays with correct types
    telescope_images = np.stack(telescope_images).astype(np.float32)
    telescope_triggers = np.array(telescope_triggers, dtype=np.int8)
    telescope_aux_inputs = np.array(telescope_aux_inputs, dtype=np.float32)

    return [telescope_images, telescope_triggers, telescope_aux_inputs,
            gamma_hadron_label]

# Data loading function for single tel HDF5 data
# Loads the image in file 'filename', in image table 'tel_type' at index 'index'
def load_data_single_tel_HDF5(filename, index, metadata, settings):

    # Load image table record from specified file and image table index
    f = return_file_handle(filename)
    tel_type = settings['processed_telescope_types'][0]
    telescope_image = load_image_HDF5(f, tel_type, index)
    if settings['crop_images']:
        telescope_image, _, _ = crop_image(telescope_image, settings)
    if settings['log_normalize_charge']:
        telescope_image[:,:,0] = np.log(telescope_image[:,:,0] - metadata['image_charge_min'][tel_type] + 1.0)

    # Get corresponding event record using event_index column
    event_index = f.root._f_get_child(tel_type)[index]['event_index']
    event_record = f.root.Event_Info[event_index]

    # Get classification label by converting CORSIKA particle code
    gamma_hadron_label = PARTICLE_ID_TO_CLASS[event_record['particle_id']]

    return [telescope_image, gamma_hadron_label]

# Return dict of auxiliary data values (currently only contains telescope position coordinates).
# Structured as auxiliary_data[telescope_positions][tel_type][tel_id] = [x,y,z]
# Checks that the same telescopes have the same position across all files.
def load_auxiliary_data_HDF5(file_list): 
    # Load telescope positions by telescope type and id
    telescope_positions = {}
    for filename in file_list:
        with tables.open_file(filename, mode='r') as f:
            # For every telescope in the file
            for row in f.root.Array_Info.iterrows():
                tel_type = row['tel_type'].decode('utf-8')
                tel_id = row['tel_id']
                if tel_type not in telescope_positions:
                    telescope_positions[tel_type] = {}
                if tel_id not in telescope_positions[tel_type]:
                        telescope_positions[tel_type][tel_id] = [row["tel_x"],
                                row["tel_y"], row["tel_z"]]
                else:
                    if telescope_positions[tel_type][tel_id] != [row["tel_x"],
                            row["tel_y"], row["tel_z"]]:
                        raise ValueError("Telescope positions do not match for telescope {} in file {}.".format(tel_id,filename))
    
    auxiliary_data = {
            'telescope_positions': telescope_positions
            }
    
    return auxiliary_data

def load_metadata_HDF5(file_list):
    num_events_by_file, particle_id_by_file , num_images_by_file = [], [], {}
    telescope_types, telescope_ids = [], {}
    image_charge_min, image_charge_max = {}, {}
    for filename in file_list:
        with tables.open_file(filename, mode='r') as f:
            num_events_by_file.append(f.root.Event_Info.shape[0])
            # Particle ID is same for all events in a given file and
            # is therefore saved in the root attributes
            particle_id_by_file.append(f.root._v_attrs.particle_type)
            # Build telescope types list and telescope ids dict for current file
            # NOTE: telescope types list is sorted in order of tel_ids
            tel_ids_types, tel_ids_types_temp = [], []
            for row in f.root.Array_Info.iterrows():
                # note: tel type strings stored in Pytables as byte strings, must be decoded
                tel_type = row['tel_type'].decode('utf-8')
                tel_id = row['tel_id']
                tel_ids_types_temp.append((tel_id,tel_type))
            # sort all (telescope id, telescope type) pairs by tel_id
            tel_ids_types_temp.sort(key=lambda i: i[0])
           
            #get max x, y, z telescope coordinates
            max_tel_x = max(row['tel_x'] for row in f.root.Array_Info.iterrows())
            max_tel_y = max(row['tel_y'] for row in f.root.Array_Info.iterrows())
            max_tel_z = max(row['tel_z'] for row in f.root.Array_Info.iterrows())

            max_telescope_pos = [max_tel_x, max_tel_y, max_tel_z]

            # Check that telescope types and ids match across all files
            if tel_ids_types != tel_ids_types_temp:
                if not tel_ids_types:
                    tel_ids_types = tel_ids_types_temp
                else:
                    raise ValueError("Telescope type/id mismatch in file {}".format(filename))
           
            # save sorted list of telescope types
            if not telescope_types:
                for tel_id, tel_type in tel_ids_types:
                    if tel_type not in telescope_types: 
                        telescope_types.append(tel_type)
            
            # save dict of telescope_ids
            if not telescope_ids:
                for tel_id, tel_type in tel_ids_types:
                    if tel_type not in telescope_ids:
                        telescope_ids[tel_type] = []
                    telescope_ids[tel_type].append(tel_id)
            
            # Save dict of number of images by tel type per telescope
            # for single tel data
            # Subtract one since index 0 corresponds to a blank template
            for tel_type in telescope_types:
                if tel_type not in num_images_by_file:
                    num_images_by_file[tel_type] = []
                num_images_by_file[tel_type].append(
                        f.root._f_get_child(tel_type).shape[0] - 1)

            # Compute dataset image max and min for normalization
            for tel_type in telescope_types:
                tel_table = f.root._f_get_child(tel_type)
                record = tel_table.read(1,tel_table.shape[0])
                images = record['image_charge']

                if tel_type not in image_charge_min:
                    image_charge_min[tel_type] = np.amin(images)
                if tel_type not in image_charge_max:
                    image_charge_max[tel_type] = np.amax(images)

                if np.amin(images) < image_charge_min[tel_type]:
                    image_charge_min[tel_type] = np.amin(images)
                if np.amax(images) > image_charge_max[tel_type]:
                    image_charge_max[tel_type] = np.amax(images)

    metadata = {
            'num_events_by_file': num_events_by_file,
            'num_telescopes': {tel_type:len(telescope_ids[tel_type]) for tel_type in telescope_types},
            'telescope_ids': telescope_ids,
            'telescope_types': telescope_types,
            'num_images_by_file': num_images_by_file,
            'particle_id_by_file': particle_id_by_file,
            'image_shapes': IMAGE_SHAPES,
            'class_to_name': CLASS_TO_NAME,
            'num_classes': len(set(particle_id_by_file)),
            'num_position_coordinates': 3,
            'image_charge_min': image_charge_min,
            'image_charge_max': image_charge_max,
            'max_telescope_pos': max_telescope_pos
            }

    return metadata

# Use the data processing settings from the user and metadata from the dataset
# to determine the final parameters of the data after processing. This is
# needed for passing to the model and for efficient data loading.
# Save the processed parameters in both dictionaries.
def add_processed_parameters(data_processing_settings, metadata):
    
    # Choose telescope types for this event. They must be available in the
    # data, chosen in the settings, and have a MAPPING_TABLE
    # NOTE: Only MSTS has a MAPPING_TABLE so far regardless of chosen types
    available_telescope_types = metadata['telescope_types']
    chosen_telescope_types = data_processing_settings['chosen_telescope_types']
    processed_telescope_types = [ttype for ttype in available_telescope_types
            if ttype in chosen_telescope_types and ttype in MAPPING_TABLES]
    
    # If single telescope mode, check that only one telescope type is enabled
    if data_processing_settings['model_type'] == 'single_tel':
        if not len(processed_telescope_types) == 1:
            raise ValueError('Exactly one telescope type must be enabled for single telescope models, number requested is: {}'.format(len(processed_telescope_types)))

    processed_parameters = {
            'processed_telescope_types': processed_telescope_types,
            'processed_image_shapes': {},
            'processed_num_telescopes': {},
            'processed_aux_input_nums': OrderedDict()
            }

    # Determine the processed image size which will be different if cropping
    for tel_type in processed_telescope_types:
        if data_processing_settings['crop_images']:
            processed_image_shape = (
                    data_processing_settings['bounding_box_size'],
                    data_processing_settings['bounding_box_size'],
                    metadata['image_shapes'][tel_type][2])
        else:
            processed_image_shape = metadata['image_shapes'][tel_type]
        processed_parameters['processed_image_shapes'][tel_type] = processed_image_shape
        processed_parameters['processed_num_telescopes'][tel_type] = metadata['num_telescopes'][tel_type]

    # Calculate the total number of auxiliary inputs
    if data_processing_settings['use_telescope_positions']:
        processed_parameters['processed_aux_input_nums']['telescope_position'] = metadata['num_position_coordinates']
    if data_processing_settings['crop_images']:
        # Image centroid x, y
        processed_parameters['processed_aux_input_nums']['shower_position'] = data_processing_settings['num_shower_coordinates']

    data_processing_settings.update(processed_parameters)
    metadata.update(processed_parameters)

def load_image_HDF5(data_file,tel_type,index):
    
    record = data_file.root._f_get_child(tel_type)[index]
    
    # Allocate empty numpy array of shape (len_trace + 1,) to hold trace plus
    # "empty" pixel at index 0 (used to fill blank areas in image)
    trace = np.empty(shape=(record['image_charge'].shape[0] + 1),dtype=np.float32)
    # Read in the trace from the record 
    trace[0] = 0.0
    trace[1:] = record['image_charge']
    
    # Create image by indexing into the trace using the mapping table, then adding a
    # dimension to given shape (length,width,1)
    telescope_image = np.expand_dims(trace[MAPPING_TABLES[tel_type]],2)
  
    return telescope_image

# Function to get all indices in each HDF5 file which pass a provided cut condition
# For single tel mode, returns all MSTS image table indices from events passing the cuts
# For array-level mode, returns all event table indices from events passing the cuts
# Cut condition must be a string formatted as a Pytables selection condition
# (i.e. for table.where()). See Pytables documentation for examples.
# If cut condition is empty, do not apply any cuts.
def apply_cuts_HDF5(file_list, cut_condition, model_type, min_num_tels=1):

    if cut_condition:
        logger.info("Cut condition: %s", cut_condition)
    else:
        logger.info("No cuts applied.")

    indices_by_file = []
    for filename in file_list:
        # No need to use the multithread-safe file open, as this function
        # is only called once
        with tables.open_file(filename, mode='r') as f:
            # For single tel, get all passing events, then collect all non-zero 
            # MSTS image indices into a flat list
            event_table = f.root.Event_Info
            if model_type == 'singletel':
                passing_events = event_table.where(cut_condition) if cut_condition else event_table.iterrows()
                rows = [row for row in passing_events if np.count_nonzero(row['MSTS_indices']) >= min_num_tels]
                indices = [i for row in rows for i in row['MSTS_indices'] if i != 0]
            # For array-level get all passing rows and return a list of all of
            # the indices
            else:
                rows = [row for row in event_table.where(cut_condition)] if cut_condition else event_table.iterrows()
                # Enforce that only events containing at least one MSTS are 
                # included. This is necessary because PyTables cut conditions
                # cannot operate on multidimensional fields.
                indices = [row.nrow for row in rows if np.count_nonzero(row['MSTS_indices']) >= min_num_tels]

        indices_by_file.append(indices)

    return indices_by_file

def split_indices_lists(indices_lists,validation_split):
    training_lists = []
    validation_lists = []
    for indices_list in indices_lists:
       num_validation = math.ceil(validation_split * len(indices_list))
       
       training_lists.append(indices_list[num_validation:len(indices_list)])
       validation_lists.append(indices_list[0:num_validation])

    return training_lists,validation_lists

# Generator function used to produce a dataset of elements (HDF5_filename,index)
# from a list of files and a list of lists of indices per file (constructed by applying cuts)
def gen_fn_HDF5(file_list,indices_by_file):
    # produce all filename,index pairs and shuffle
    filename_index_pairs = [(filename,i) for (filename, indices_list) in zip(file_list,indices_by_file) for i in indices_list]
    random.shuffle(filename_index_pairs)

    for (filename,i) in filename_index_pairs:
        yield (filename.encode('utf-8'),i)

def get_data_generators_HDF5(file_list, metadata, settings, mode='train'):

    # Get number of examples by file
    if settings['model_type'] == 'singletel': # get number of images
        telescope_type = settings['processed_telescope_types'][0]
        num_examples_by_file = metadata['num_images_by_file'][telescope_type]
    else: # get number of events
        num_examples_by_file = metadata['num_events_by_file']

    # Log general information on dataset based on metadata dictionary
    logger.info("%d data files read.", len(file_list))
    logger.info("Telescopes in data:")
    for tel_type in metadata['telescope_ids']:
        logger.info(tel_type + ": "+'[%s]' % ', '.join(map(str,metadata['telescope_ids'][tel_type]))) 
    
    num_examples_by_label = {}
    for i,num_examples in enumerate(num_examples_by_file):
        particle_id = metadata['particle_id_by_file'][i]
        if particle_id not in num_examples_by_label: num_examples_by_label[particle_id] = 0
        num_examples_by_label[particle_id] += num_examples

    total_num_examples = sum(num_examples_by_label.values())

    logger.info("%d total examples.", total_num_examples)
    logger.info("Num examples by label:")
    for label in num_examples_by_label:
        logger.info("%s: %d (%f%%)", label, num_examples_by_label[label], 100 * float(num_examples_by_label[label])/total_num_examples)

    # Apply cuts
    indices_by_file = apply_cuts_HDF5(file_list, settings['cut_condition'], settings['model_type'], min_num_tels=settings['min_num_tels'])

    # Log info on cuts
    num_passing_examples_by_label = {}
    for i,index_list in enumerate(indices_by_file):
        num_passing_examples = len(index_list)
        particle_id = metadata['particle_id_by_file'][i]
        if particle_id not in num_passing_examples_by_label:
            num_passing_examples_by_label[particle_id] = 0
        num_passing_examples_by_label[particle_id] += num_passing_examples

    num_passing_examples = sum(num_passing_examples_by_label.values())

    logger.info("%d total examples passing cuts.", num_passing_examples)
    logger.info("Num examples by label:")
    for label in num_passing_examples_by_label:
        logger.info("%s: %d (%f%%)", label, num_passing_examples_by_label[label], 100 * float(num_passing_examples_by_label[label])/num_passing_examples)

    # Add post-cut computed class weights to metadata dictionary
    metadata['class_weights'] = [] 
    for particle_id in sorted(num_passing_examples_by_label,key=lambda x: PARTICLE_ID_TO_CLASS[x]):
        metadata['class_weights'].append(num_passing_examples/float(num_passing_examples_by_label[particle_id]))

    if mode == 'train':
        # Split indices lists into training and validation
        training_indices, validation_indices = split_indices_lists(indices_by_file,
                settings['validation_split'])

        def training_generator():
            return gen_fn_HDF5(file_list,training_indices)
        def validation_generator():
            return gen_fn_HDF5(file_list,validation_indices)

        return training_generator, validation_generator

    elif mode == 'test':

        return indices_by_file

# Crop an image about the shower center, optionally applying image cleaning
# to obtain a better fit. The shower centroid is calculated as the mean of
# pixel positions weighted by the charge, after cleaning. The cropped image is
# obtained as a square bounding box centered on the centroid of side length
# bounding_box_size.
def crop_image(image, settings):

    # Apply image cleaning
    image_cleaning_method = settings['image_cleaning_method']
    if image_cleaning_method == "none":
        # Don't apply any cleaning
        cleaned_image = image
    elif image_cleaning_method == "twolevel":
        # Apply two-level cleaning to isolate the shower. First, filter for
        # shower pixels by applying a high charge cut (picture threshold).
        # Next, retain weaker pixels at the shower edge by allowing pixels
        # adjacent to those passing the first cut to pass a weaker cut
        # (boundary threshold).
        
        # Get only the first channel (charge) of an image of arbitrary depth
        image_charge = image[:,:,0]

        # Apply picture threshold to charge image to get mask
        m = (image_charge > settings['picture_threshold']).astype(np.uint8)
        # Dilate the mask once to add all adjacent pixels (i.e. kernel is 3x3)
        kernel = np.ones((3,3), np.uint8) 
        m = cv2.dilate(m, kernel)
        # Apply boundary threshold to keep weaker but adjacent pixels
        m = (m * image_charge > settings['boundary_threshold']).astype(np.uint8)
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
    bounding_box_size = settings['bounding_box_size']
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
    returned_image = (cleaned_image if settings['return_cleaned_images'] else image)
    cropped_image[x_min_cropped:x_max_cropped+1,y_min_cropped:y_max_cropped+1,:] = returned_image[x_min_image:x_max_image+1,y_min_image:y_max_image+1,:]

    return cropped_image, x_0, y_0

