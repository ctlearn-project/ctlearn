from operator import itemgetter
import threading
import logging
import math

import tables
import numpy as np

from ctalearn.image import MAPPING_TABLES, IMAGE_SHAPES

logger = logging.getLogger(__name__)

# Multithread-safe PyTables open and close file functions
# See http://www.pytables.org/latest/cookbook/threading.html
lock = threading.Lock()

def synchronized_open_file(*args, **kwargs):
    with lock:
        return tables.open_file(*args, **kwargs)

def synchronized_close_file(self, *args, **kwargs):
    with lock:
        return self.close(*args, **kwargs)

# Data loading function for event-wise (array-level) HDF5 data loading
def load_data_eventwise_HDF5(filename, index, auxiliary_data, metadata,sort_telescopes_by_trigger=False):

    # Read the event record for the given filename and index
    f = synchronized_open_file(filename.decode('utf-8'), mode='r')
    record = f.root.Event_Info[index]
    
    # Get classification label by converting CORSIKA particle code
    if record['particle_id'] == 0: # gamma ray
        gamma_hadron_label = 1
    elif record['particle_id']  == 101: # proton
        gamma_hadron_label = 0
    else:
        raise ValueError("Unimplemented particle_id value: {}".format(record['particle_id']))
  
    # Collect image indices (indices into the image tables)
    # for each telescope type in this event
    telescope_types = metadata['telescope_types']
    image_indices = {tel_type:record[tel_type+"_indices"] for tel_type in telescope_types}
   
    # Collect images and binary trigger values
    telescope_images = []
    telescope_triggers = []
    for tel_type in telescope_types:
        # Only save MSTS (other telescope types do not have implemented MAPPING_TABLES yet)
        if tel_type in MAPPING_TABLES:
            for i in image_indices[tel_type]:
                if i == 0:
                    # Telescope did not trigger. Its outputs will be dropped
                    # out, so input is arbitrary. Use an empty array for
                    # efficiency.
                    telescope_images.append(np.empty(metadata['image_shapes'][tel_type]))
                    telescope_triggers.append(0)
                else:
                    telescope_image = load_image_HDF5(f,tel_type,i)
                    telescope_images.append(telescope_image)
                    telescope_triggers.append(1)
    
    synchronized_close_file(f)

    # Collect telescope positions from auxiliary data
    # telescope_positions is a list of lists ex. [[x1,y1,z1],[x2,y2,z2],...]
    telescope_positions = []
    for tel_type in telescope_types:
        if tel_type in MAPPING_TABLES:
            for tel_id in sorted(auxiliary_data['telescope_positions'][tel_type].keys):
                telescope_positions.append(auxiliary_data['telescope_positions'][tel_type][tel_id])

    if sort_telescopes_by_trigger:
        # Sort the images, triggers, and grouped positions by trigger, listing
        # the triggered telescopes first
        telescope_images, telescope_triggers, telescope_positions = map(list,
                zip(*sorted(zip(telescope_images, telescope_triggers,
                    telescope_positions), reverse=True, key=itemgetter(1))))

    # Convert to numpy arrays with correct types
    telescope_images = np.stack(telescope_images).astype(np.float32)
    telescope_triggers = np.array(telescope_triggers, dtype=np.int8)
    telescope_positions = np.array(telescope_positions, dtype=np.float32)

    return [telescope_images, telescope_triggers, telescope_positions,
            gamma_hadron_label]

# Data loading function for single tel HDF5 data
# Loads the image in file 'filename', in image table 'tel_type' at index 'index'
def load_data_single_tel_HDF5(filename, tel_type, index, metadata):

    # Load image table record from specified file and image table index
    f = synchronized_open_file(filename.decode('utf-8'), mode='r')
    telescope_image = load_image_HDF5(f,tel_type,index)

    # Get corresponding event record using event_index column
    event_index = f.root._f_get_child(tel_type)[index]['event_index']
    event_record = f.root.Event_Info[event_index]

    # Get classification label by converting CORSIKA particle code
    if event_record['particle_id'] == 0: # gamma ray
        gamma_hadron_label = 1
    elif event_record['particle_id'] == 101: # proton
        gamma_hadron_label = 0
    else:
        raise ValueError("Unimplemented particle_id value: {}".format(record['particle_id']))
  
    synchronized_close_file(f)

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
            for row in f.root.Telescope_Info.iterrows():
                tel_type = row['tel_type'].decode('utf-8')
                tel_id = row['tel_id']
                if tel_type not in telescope_positions:
                    telescope_positions[tel_type] = []
                if tel_id not in telescope_positions[tel_type]:
                        telescope_positions[tel_type][tel_id] = [row["tel_x"],row["tel_y"],row["tel_z"]]
                else:
                    if telescope_positions[tel_type][tel_id] != [row["tel_x"],row["tel_y"],row["tel_z"]]:
                        raise ValueError("Telescope positions do not match for telescope {} in file {}.".format(tel_id,filename))
    
    auxiliary_data = {
            'telescope_positions': telescope_positions
            }
    
    return auxiliary_data

def load_metadata_HDF5(file_list):
    num_events_by_file = []
    num_images_by_file = {}
    particle_id_by_file = []
    telescope_types = []
    telescope_ids = {}
    image_charge_max = {}
    image_charge_min = {}
    for filename in file_list:
        with tables.open_file(filename, mode='r') as f:
            # Number of events
            num_events_by_file.append(f.root.Event_Info.shape[0])
            # Particle ID (same for all events in a single file)
            particle_id_by_file.append(f.root._v_attrs.particle_type)
            # Build telescope types list and telescope ids dict for current file
            # NOTE: telescope types list is sorted in order of tel_ids
            telescope_types_ids = []
            telescope_types_current_file = []
            telescope_ids_current_file = {}
            for row in f.root.Telescope_Info.iterrows():
                tel_type = row['tel_type'].decode('utf-8')
                tel_id = row['tel_id']
                telescope_types_ids.append((tel_id,tel_type))
            for tel_id,tel_type in sorted(telescope_types_ids,key=lambda i: i[0]):
                if tel_type not in telescope_types_current_file:
                    telescope_types_current_file.append(tel_type)
                if tel_type not in telescope_ids_current_file:
                    telescope_ids_current_file[tel_type] = []
                telescope_ids_current_file[tel_type].append(tel_id)

            # Check that telescope types and ids match across all files
            if not telescope_types:
                telescope_types = telescope_types_current_file
            else:
                if telescope_types != telescope_types_current_file:
                    raise ValueError("Telescope type mismatch in file {}".format(filename))

            if not telescope_ids:
                telescope_ids = telescope_ids_current_file
            else:
                for tel_type in telescope_types:
                    if telescope_ids[tel_type] != telescope_ids_current_file[tel_type]:
                        raise ValueError("Telescope id mismatch in file {} (tel_type {})".format(filename,tel_type))

            # Number of images per telescope (for single tel data)
            for tel_type in telescope_types:
                if tel_type not in num_images_by_file:
                    num_images_by_file[tel_type] = []
                num_images_by_file[tel_type].append(f.root._f_get_child(tel_type).shape[0])

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
            'num_classes': len(set(particle_id_by_file)),
            'num_position_coordinates': 3,
            'image_charge_min': image_charge_min,
            'image_charge_max': image_charge_max
            }

    return metadata

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
def apply_cuts_HDF5(file_list, cut_condition, model_type):

    if cut_condition:
        logger.info("Cut condition: {}".format(cut_condition))
    else:
        logger.info("No cuts applied.")

    indices_by_file = []
    for filename in file_list:
        # No need to use the multithread-safe file open, as this function
        # is only called once
        with tables.open_file(filename, mode='r') as f:
            # For single tel, get all passing events, then collect all non-zero 
            # MSTS image indices into a flat list
            if model_type == 'singletel':
                passing_events = f.root.Event_Info.where(cut_condition) if cut_condition else f.root.Event_Info.iterrows()
                indices = [i for row in passing_events for i in row['MSTS_indices'] if i != 0]
            # For array-level get all passing rows and return a list of all of
            # the indices
            else:
                indices = [row.nrow for row in table.where(cut_condition)] if cut_condition else range(table.nrows)

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
    for filename,indices_list in zip(file_list,indices_by_file):
        for i in indices_list:
            yield (filename.encode('utf-8'),i)

def get_data_generators_HDF5(file_list,cut_condition,model_type,validation_split=0.1):

    metadata = load_metadata_HDF5(file_list)

    # Get number of examples by file (for single tel, number of MSTS images, for array-level, number of events)
    num_examples_by_file = metadata['num_images_by_file']['MSTS'] if model_type == 'singletel' else metadata['num_events_by_file']

    # Log general information on dataset based on metadata dictionary
    logger.info("{} data files read.".format(len(file_list)))
    logger.info("Telescopes in data:")
    for tel_type in metadata['telescope_ids']:
        logger.info(tel_type + ": "+'[%s]' % ', '.join(map(str,metadata['telescope_ids'][tel_type]))) 
    
    num_examples_by_label = {}
    for i,num_examples in enumerate(num_examples_by_file):
        particle_id = metadata['particle_id_by_file'][i]
        if particle_id not in num_examples_by_label: num_examples_by_label[particle_id] = 0
        num_examples_by_label[particle_id] += num_examples

    total_num_examples = sum(num_examples_by_label.values())

    logger.info("{} total examples.".format(total_num_examples))
    logger.info("Num examples by label:")
    for label in num_examples_by_label:
        logger.info("{}: {} ({}%)".format(label,num_examples_by_label[label], 100 * float(num_examples_by_label[label])/total_num_examples))

    # Apply cuts
    indices_by_file = apply_cuts_HDF5(file_list,cut_condition,model_type)

    # Log info on cuts
    num_passing_examples_by_label = {}
    for i,index_list in enumerate(indices_by_file):
        num_passing_examples = len(index_list)
        particle_id = metadata['particle_id_by_file'][i]
        if particle_id not in num_passing_examples_by_label:
            num_passing_examples_by_label[particle_id] = 0
        num_passing_examples_by_label[particle_id] += num_passing_examples

    num_passing_examples = sum(num_passing_examples_by_label.values())
    num_validation_examples = int(validation_split * num_passing_examples)
    num_training_examples = num_passing_examples - num_validation_examples

    logger.info("{} total examples passing cuts.".format(num_passing_examples))
    logger.info("Num examples by label:")
    for label in num_passing_examples_by_label:
        logger.info("{}: {} ({}%)".format(label,num_passing_examples_by_label[label], 100 * float(num_passing_examples_by_label[label])/num_passing_examples))

    # Split indices lists into training and validation
    training_indices, validation_indices = split_indices_lists(indices_by_file,validation_split)

    def training_generator():
        return gen_fn_HDF5(file_list,training_indices)
    def validation_generator():
        return gen_fn_HDF5(file_list,validation_indices)

    return training_generator, validation_generator
