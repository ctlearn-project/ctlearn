from operator import itemgetter
import threading
import math
from collections import OrderedDict
import random
from abc import ABC, abstractmethod

import tables
import numpy as np

from ctalearn.image import MAPPING_TABLES, IMAGE_SHAPES

# Maps CORSIKA particle id codes
# to particle class names
PARTICLE_ID_TO_NAME = {
        0: 'gamma',
        101:'proton'
        } 

# Abstract class representing a CTALearn dataset (collection of shower events,
# event parameters, images, and corresponding metadata).
# Provided as a template for the implementation of alternative data formats 
# for storing training data.
class CTALearnDataset(ABC):

    @abstractmethod
    def get_image(self, run_id, event_id, tel_id):
        pass

    @abstractmethod
    def get_event_example(self, run_id, event_id):
        pass

    @abstractmethod
    def get_single_tel_example(self, run_id, event_id, telescope_id):
        pass
    
    # return a standard collection of metadata parameters describing the data
    @abstractmethod
    def get_metadata(self):
        pass

    # return a dictionary of auxiliary data
    @abstractmethod
    def get_auxiliary_data(self):
        pass

    @abstractmethod
    def get_example_generators(self, example_type, cut_condition="", mode='train', validation_split=0.1):
        pass

# PyTables HDF5 implementation of CTALearn_Dataset
# Corresponds to standard CTA ML format specified by
# ImageExtractor (https://github.com/cta-observatory/image-extractor).
class HDF5Dataset(CTALearnDataset):
    
    def __init__(self, file_list):

        # construct dict of filename:file_handle pairs 
        # but do not open file handles until required by 
        # get_file_handle
        self.files = {filename:None for filename in file_list}

        # Load metadata and auxiliary parameters
        # corresponding to the dataset
        self.load_metadata()
        self.load_auxiliary_data()

        # By default, select all telescope types with mapping tables
        tel_types = [tel_type for tel_type in self.telescopes 
                if tel_type in MAPPING_TABLES]
        self.select_telescopes(tel_types=tel_types)

    # Multithread-safe PyTables open and close file functions
    # See http://www.pytables.org/latest/cookbook/threading.html
    lock = threading.Lock()

    @staticmethod
    def synchronized_open_file(*args, **kwargs):
        with lock:
            return tables.open_file(*args, **kwargs)

    @staticmethod
    def synchronized_close_file(self, *args, **kwargs):
        with lock:
            return self.close(*args, **kwargs)

    # Store/retrieve the file handles corresponding to each filename.
    # Maintaining open file handles avoids the need to close and re-open
    # the underlying HDF5 files when reading events/images
    def get_file_handle(self, filename):
        if self.files[filename] is None:
            self.files[filename] = synchronized_open_file(
                    filename.decode('utf-8'), mode='r')
        return self.files[filename]

    # Compute and save a collection of metadata parameters
    # which describe the dataset
    def load_metadata(self):

        self.particle_ids = {}

        # OrderedDict with telescope types as keys and list of telescope ids
        # of each type (in sorted order) as values
        # NOTE: the telescope types are ordered by increasing telescope id
        self.telescopes = OrderedDict()

        self.events = [] 
        self.images = {}

        self.num_events = 0
        self.num_images = {}

        self.num_events_by_particle_id = {}
        self.num_images_by_particle_id = {}

        self.num_position_coordinates = 3
        self.telescope_positions = {}
        self.max_telescope_positions = {}
 
        self.image_charge_mins = {}
        self.image_charge_maxes = {}
       
        self.__events_to_indices = {}
        self.__single_tel_events_to_indices = {}

        self.__tel_id_to_type_location = {}

        for filename in self.files:
            with tables.open_file(filename.decode('utf-8'), mode='r') as f:
                # Particle ID is same for all events in a given file and
                # is therefore saved in the root attributes
                particle_id = f.root._v_attrs.particle_type
                self.particle_ids.add(particle_id)

                tel_ids_types = []
                for row in f.root.Array_Info.iterrows():
                    # note: tel type strings stored in Pytables as byte strings, must be decoded
                    tel_type = row['tel_type'].decode('utf-8')
                    tel_id = row['tel_id']
                    tel_ids_types.append((tel_id,tel_type))
                    if tel_type not in self.telescope_positions:
                            self.telescope_positions[tel_type] = {}
                        if tel_id not in self.telescope_positions[tel_type]:
                                self.telescope_positions[tel_type][tel_id] = [row["tel_x"],
                                        row["tel_y"], row["tel_z"]]
                        else:
                            if self.telescope_positions[tel_type][tel_id] != [row["tel_x"],
                                    row["tel_y"], row["tel_z"]]:
                                raise ValueError("Telescope positions do not match for telescope {} in file {}.".format(tel_id,filename))
     
                # sort all (telescope id, telescope type) pairs by tel_id
                tel_ids_types.sort(key=lambda i: i[0])

                # For every telescope in the file
                for row in f.root.Array_Info.iterrows():
                    tel_type = row['tel_type'].decode('utf-8')
                    tel_id = row['tel_id']
                    
                telescopes = OrderedDict()
                index = 0
                prev_tel_type = tel_ids_types[0][1]
                for tel_id, tel_type in tel_ids_types:
                    if tel_type not in telescopes:
                        telescopes[tel_type] = []
                    telescopes[tel_type].append(tel_id)
                    self.__tel_id_to_type_location[tel_id] = (tel_type,index)
                    if tel_type != prev_tel_type:
                        index = 0
                    else:
                        index += 1
                    prev_tel_type = tel_type

                if not self.telescopes:
                    self.telescopes = telescopes
                else:
                    if self.telescopes != telescopes:
                        raise ValueError("Telescope type/id mismatch in file {}".format(filename))

                # Compute max x, y, z telescope coordinates for normalization
                max_tel_x = max(row['tel_x'] for row in f.root.Array_Info.iterrows())
                max_tel_y = max(row['tel_y'] for row in f.root.Array_Info.iterrows())
                max_tel_z = max(row['tel_z'] for row in f.root.Array_Info.iterrows())

                self.max_telescope_pos = [max_tel_x, max_tel_y, max_tel_z]
                
                for row in f.root.Event_Info.iterrows():
                    if particle_id not in self.num_events_by_particle_id:
                        self.num_events_by_particle_id[particle_id] = 0

                    self.events.append((row['run_id'],row['event_id']))
                    self.__events_to_indices[(row['run_id'],row['event_id'])] = (filename, row.nrow)

                    self.num_events_by_particle_id[particle_id] += 1
                    self.num_events += 1

                    for tel_type in self.telescopes:
                        tel_ids = self.telescopes[tel_type]
                        indices = row[tel_type + '_indices']
                        if not self.num_images[tel_type]:
                            self.num_images[tel_type] = 0
                        if not self.images[tel_type]:
                            self.images[tel_type] = []
                        for tel_id, image_index in zip(tel_ids, indices):
                            if image_index != 0:
                                self.images[tel_type].append((row['run_id'], row['event_id'], tel_id))
                                self.__single_tel_events_to_indices[(row['run_id'], row['event_id'], tel_id)] = (filename, tel_type, image_index)
                                self.num_images[tel_type] += 1
                                self.num_images_by_particle_id[tel_type][particle_id] += 1

                # Compute max and min pixel value in each telescope image
                # type for normalization
                # NOTE: This step is time-intensive.
                for tel_type in self.telescopes.keys():
                    tel_table = f.root._f_get_child(tel_type)
                    records = tel_table.read(1,tel_table.shape[0])
                    images = records['image_charge']

                    if tel_type not in image_charge_mins:
                        image_charge_mins[tel_type] = np.amin(images)
                    if tel_type not in image_charge_maxes:
                        image_charge_maxes[tel_type] = np.amax(images)

                    if np.amin(images) < image_charge_mins[tel_type]:
                        image_charge_mins[tel_type] = np.amin(images)
                    if np.amax(images) > image_charge_maxes[tel_type]:
                        image_charge_maxes[tel_type] = np.amax(images)
            
            # create mapping from particle ids to labels
            # and from labels to names
            self.ids_to_labels = {particle_id:i 
                    for i, particle_id in enumerate(list(self.particle_ids).sort())}
            self.labels_to_names = {i:PARTICLE_ID_TO_NAME[particle_id] 
                    for particle_id, i in self.labels.items()}

            # By default, all telescopes with mapping tables will be selected
            self.selected_telescopes = {tel_type:self.telescopes[tel_type] 
                    for tel_type in self.telescopes 
                    if tel_type in MAPPING_TABLES}

    # Method returning a dict of selected metadata parameters
    def get_metadata(self):

        metadata = {
                'particle_ids': self.particle_ids,
                'telescopes': self.telescopes,
                'num_events_by_particle_id': self.num_events_by_particle_id,
                'num_images_by_particle_id': self.num_images_by_particle_id,
                'num_position_coordinates': self.num_position_coordinates,
           }

        return metadata

    # Return dictionary of auxiliary data.
    def get_auxiliary_data(self):
        auxiliary_data = {
            'telescope_positions': self.telescope_positions
            }

        return auxiliary_data

    # Select which telescopes from the full dataset to include in each event 
    # by suppling either a list of telescope types or a list of telescope ids.
    # Can be called multiple times to update/overwrite the selected telescopes.
    def select_telescopes(self, tel_types=None, tel_ids=None):
       
        self.selected_telescopes = OrderedDict()

        if tel_types and tel_ids:
            raise ValueError("Both tel_types and tel_ids provided.
                Provide only one or the other.")
        elif not tel_types and not tel_ids:
            raise ValueError("One of tel_types and tel_ids must be provided.")
        elif tel_types:
            allowed_tel_types = []
            for tel_type in tel_types:
                if tel_type not in self.telescopes:
                    raise ValueError("Selected tel type {} not found in dataset.".format(tel_type))
                elif tel_type not in MAPPING_TABLES:
                    raise NotImplementedError("Mapping table for selected tel type {} not implemented.".format(tel_type))
                else:
                    allowed_tel_types.append(tel_type)
            for tel_type in self.telescopes:
                if tel_type in allowed_tel_types:
                    self.selected_telescopes[tel_type] = self.telescopes[tel_type]
        elif tel_ids:
            # get tel_ids and types for all telescopes
            all_tel_ids = []
            for tel_type in self.telescopes:
                for tel_id in self.telescopes[tel_type]:
                    all_tel_ids[tel_id] = tel_type
            tel_ids.sort()
            for tel_id in tel_ids:
                if tel_id not in all_tel_ids:
                    raise ValueError("Selected tel id {} not found in dataset.".format(tel_id))
                elif all_tel_ids[tel_id] not in MAPPING_TABLES:
                    raise NotImplementedError(
                            "Mapping table for tel type {} of selected tel id {} not implemented.".format(
                                all_tel_ids[tel_id],tel_id))
                else:
                    if all_tel_ids[tel_id] not in self.selected_telescopes:
                        self.selected_telescopes = []
                    self.selected_telescopes[all_tel_ids[tel_id]].append(tel_id)
  
    # Get a single telescope image from a particular event, 
    # uniquely identified by a tuple (run_id, event_id, tel_id).
    # The raw 1D trace is transformed into a 1-channel 2D image using a
    # mapping table but no other processing is done.
    def get_image(self, run_id, event_id, tel_id):
        
        tel_type, _ = self.__tel_id_to_type_location[tel_id]
        if tel_type not in MAPPING_TABLES:
            raise ValueError("Requested image from tel_type {} without valid mapping table.".format(tel_type))

        # get filename, image table name (telescope type), and index
        # corresponding to the desired image
        filename, tel_type, index = self.__single_tel_events_to_indices[(run_id, event_id, tel_id)]
        
        f = self.get_file_handle(filename)
        record = f.root._f_get_child(tel_type)[index]
        
        # Allocate empty numpy array of shape (len_trace + 1,) to hold trace plus
        # "empty" pixel at index 0 (used to fill blank areas in image)
        trace = np.empty(shape=(record['image_charge'].shape[0] + 1),dtype=np.float32)
        # Read in the trace from the record 
        trace[0] = 0.0
        trace[1:] = record['image_charge']
        
        # Create image by indexing into the trace using the mapping table, then adding a
        # dimension to given shape (length,width,1)
        image = trace[MAPPING_TABLES[tel_type]]
        image = np.expand_dims(image, 2)

        return image

    def get_single_tel_example(self, run_id, event_id, tel_id):
        
        # get image from image table
        image = self.get_image(run_id, event_id, tel_id) 

        # locate corresponding event record to get particle type
        filename, index = self.__events_to_locations[(run_id, event_id)]
        f = self.get_file_handle(filename)
        event_record = f.root.Event_Info[index]

        # Get classification label by converting CORSIKA particle code
        label = self.ids_to_labels[event_record['particle_id']] 

        return [image, label]

    def get_event_example(self, run_id, event_id, use_telescope_positions=True):

        # get filename, image table name (telescope type), and index
        # corresponding to the desired image
        filename, index = self.__events_to_locations[(run_id, event_id)]
        
        f = self.get_file_handle(filename)
        record = f.root._f_get_child(tel_type)[index]

        # Get classification label by converting CORSIKA particle code
        label = self.ids_to_labels[event_record['particle_id']] 
      
        # Collect images and binary trigger values only for telescopes
        # in selected_telescopes
        data = {}

        for tel_type in self.selected_telescopes:
            images = []
            triggers = []
            image_shape = IMAGE_SHAPES[tel_type] 
            for tel_id in self.selected_telescopes[tel_type]:
                _, location = self.__tel_id_to_type_location[tel_id]
                i = record[tel_type + "_indices"][location]
                if i == 0:
                    # Telescope did not trigger. Its outputs will be dropped
                    # out, so input is arbitrary. Use an empty array for
                    # efficiency.
                    images.append(np.empty(image_shape))
                    triggers.append(0)  
                else:
                    image = self.get_image(run_id, event_id, tel_id) 
                    images.append(image)
                    triggers.append(1)
                if use_telescope_positions:
                    telescope_position = self.telescope_positions[tel_type][tel_id]
                    telescope_position = [float(telescope_position[i]) / self.max_telescope_position[i] 
                            for i in range(self.num_position_coordinates)]
                    aux_inputs.append(telescope_position)
            
            data[tel_type] = [images, triggers, aux_inputs]
        
        return [data, label]

    # Function to get all indices in each HDF5 file which pass a provided cut condition
    # For single tel mode, returns all MSTS image table indices from events passing the cuts
    # For array-level mode, returns all event table indices from events passing the cuts
    # Cut condition must be a string formatted as a Pytables selection condition
    # (i.e. for table.where()). See Pytables documentation for examples.
    # If cut condition is empty, do not apply any cuts.

    # Min num tels is a dictionary specifying the minimum number of telescopes of each type required
    def apply_cuts(self,
        example_type,
        cut_condition="",
        min_num_tels=None):

        if min_num_tels is None:
            min_num_tels = {'MSTS': 1}

        examples_list = []
        num_examples_by_particle_id = {}

        for filename in self.files:
            particle_id = f.root._v_attrs.particle_type
            if particle_id not in num_examples_by_particle_id:
                num_examples_by_particle_id[particle_id] = 0
            with tables.open_file(filename, mode='r') as f:
                event_table = f.root.Event_Info
                rows = [row for row in event_table.where(cut_condition)] if cut_condition else event_table.iterrows()
                for row in rows:
                    # First check if min num tels cut is passed
                    has_min_num_tels = True
                    for tel_type in self.selected_telescopes:
                        if tel_type in min_num_tels and np.count_nonzero(row[tel_type + "_indices"] < min_num_tels[tel_type]:
                            has_min_num_tels = False
                    if not has_min_num_tels:
                        pass
                    
                    # If example_type is a tel_type, add all images
                    # from selected telescopes of that type
                    if example_type in self.selected_telescopes: 
                        for tel_id in self.selected_telescopes[example_type]:
                            examples_list.append((row["run_id"], row["event_id"], row["tel_id"]))
                            num_examples_by_particle_id[particle_id] += 1
                    # if example type is 
                    elif example_type == "event":                               
                        examples_list.append((row["run_id"], row["event_id"]))
                        num_examples_by_particle_id[particle_id] += 1

        # get total number of examples
        num_examples = 0
        for particle_id in num_examples_by_particle_id:
            num_examples += num_examples_by_particle_id[particle_id]

        # compute class weights
        class_weights = []
        for particle_id in sorted(num_examples_by_particle_id, key=lambda x: self.ids_to_labels[x]):
            class_weights.append(num_examples/float(num_examples_by_particle_id[particle_id]))

        return examples_list, num_examples_by_particle_id, class_weights
 
    # Given a list of examples (tuples), returns a generator function 
    # which yields from the list. Optionally shuffles the examples
    @staticmethod
    def get_generator_function(examples_list, shuffle=True):

        def generator_fn():
            if shuffle:
                random.shuffle(examples_list)
            for example in examples_list:
                yield example

        return generator_fn

    def get_example_generators(self, 
        example_type,
        cut_condition="",
        mode='train',
        validation_split=0.1):

        # Apply cuts to get list of valid examples (either array-level
        # events or single-tel examples)
        examples, num_examples_by_id, class_weights = self.apply_cuts(mode, example_type, cut_condition, min_num_tels)

        # log class breakdown after cuts
        log_class_breakdown(num_examples_by_id)

        if mode == 'train':
            # Split examples into training and validation sets
            num_validation = math.ceil(validation_split * len(examples))
           
            training_examples = examples[num_validation:len(examples)])
            validation_examples = examples[0:num_validation]

            # Convert lists of training and validation examples into generators
            training_generator_fn = self.get_generator_fn(training_examples)
            validation_generator_fn = self.get_generator_fn(validation_examples)

            return training_generator_fn, validation_generator_fn, class_weights

        elif mode == 'test':

            test_generator_fn = self.get_generator_fn(examples)

            return test_generator_fn, class_weights

# given a dictionary of form {particle_id: num_examples}
# logs an informative message about the proportions belonging
# to different particle ids.
def log_class_breakdown(num_by_particle_id, logger=None):

    if not logger: logger = logging.get_logger()

    total_num = sum(num_by_particle_id.values())
    logger.info("%d total.", total_num)
    for particle_id in num_by_particle_id:
        logger.info("%d: %d (%f%%)",
                particle_id, 
                num_by_particle_id[particle_id], 
                100 * float(num_by_particle_id[particle_id])/total_num)
                )

