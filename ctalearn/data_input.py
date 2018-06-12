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

# General abstract class for loading CTA event data from a dataset
# stored in some file format.
# Provided as a template for the implementation of alternative data formats 
# for storing training data.
class DataLoader(ABC):

    @abstractmethod
    def get_image(self):
        pass

    @abstractmethod
    def get_example(self):
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
    def get_example_generators(self):
        pass

# PyTables HDF5 implementation of DataLoader
# Corresponds to standard CTA ML format specified by
# ImageExtractor (https://github.com/cta-observatory/image-extractor).
class HDF5DataLoader(DataLoader):

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

    def __init__(self, 
            file_list,
            mode="train",
            example_type="single_tel",
            selected_tel_types=['MSTS'],
            selected_tel_ids=None,
            min_num_tels={'MSTS':1},
            cut_condition="",
            validation_split=0.1,
            data_processor=None,
            seed=None
            ):

        # construct dict of filename:file_handle pairs 
        self.files = {filename:synchronized_open_file(filename.decode('utf-8'), mode='r')
                    for filename in file_list}

        # Data loading settings
        self.mode = mode
        self.example_type = example_type
        self.cut_condition = cut_condition
        self.validation_split = validation_split
        self.data_processor = data_processor
        self.seed = seed

        # Compute and save metadata describing dataset
        self._load_metadata()
        
        # Select desired telescopes
        self._select_telescopes(tel_types=selected_tel_types, tel_ids=selected_tel_ids)

        # Apply cuts to get lists of valid examples
        self._apply_cuts(self.cut_condition)

    # Compute and save a collection of metadata parameters
    # which describe the dataset
    def _load_metadata(self):

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
        self.__single_tel_examples_to_indices = {}

        self.__tel_id_to_type_location = {}

        for filename in self.files:
            # get file handle
            f = self.files[filename]
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
                            self.__single_tel_examples_to_indices[(row['run_id'], row['event_id'], tel_id)] = (filename, tel_type, image_index)
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
    def _select_telescopes(self, tel_types=None, tel_ids=None):
       
        self.selected_telescopes = OrderedDict()

        if tel_types and tel_ids:
            raise ValueError("Both tel_types and tel_ids provided.
                Provide only one or the other.")
        elif not tel_types and not tel_ids:
            raise ValueError("One of tel_types and tel_ids must be provided.")
        elif tel_types:
            if len(tel_types) > 1 and self.example_type == "single_tel":
                raise ValueError("Cannot select multiple telescope types in single tel mode.")
            
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
            if len(set([all_tel_ids[tel_id] for tel_id in tel_ids])) > 1 and self.example_type == "single_tel":
                raise ValueError("Cannot select telescopes of multiple types in single tel mode.")
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
        filename, tel_type, index = self.__single_tel_examples_to_indices[(run_id, event_id, tel_id)]
        
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

    def get_example(self, index):

        if self.example_type == "single_tel":

            # get image from image table
            image = self.get_image(run_id, event_id, tel_id) 

            # locate corresponding event record to get particle type
            filename, index = self.__events_to_locations[(run_id, event_id)]
            f = self.get_file_handle(filename)
            event_record = f.root.Event_Info[index]

            # Get classification label by converting CORSIKA particle code
            label = self.ids_to_labels[event_record['particle_id']] 

            data = {self.selected_telescopes.keys()[0]:image}

        elif self.example_type == "array":
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
 
        if self.data_processor:
            data, label = self.data_processor.process_example(data, label)

        return [data, label]

    # Function to get all indices in each HDF5 file which pass a provided cut condition
    # For single tel mode, returns all MSTS image table indices from events passing the cuts
    # For array-level mode, returns all event table indices from events passing the cuts
    # Cut condition must be a string formatted as a Pytables selection condition
    # (i.e. for table.where()). See Pytables documentation for examples.
    # If cut condition is empty, do not apply any cuts.

    # Min num tels is a dictionary specifying the minimum number of telescopes of each type required
    def _apply_cuts(self):

        passing_examples = []
        self.passing_num_examples_by_particle_id = {}

        for filename in self.files:
            f = self.files[filename]
            
            particle_id = f.root._v_attrs.particle_type
            if particle_id not in self.passing_num_examples_by_particle_id:
                self.passing_num_examples_by_particle_id[particle_id] = 0
            
            event_table = f.root.Event_Info
            rows = [row for row in event_table.where(cut_condition)] if cut_condition else event_table.iterrows()
            for row in rows:
                # First check if min num tels cut is passed
                has_min_num_tels = True
                for tel_type in self.selected_telescopes:
                    if tel_type in self.min_num_tels and np.count_nonzero(row[tel_type + "_indices"] < self.min_num_tels[tel_type]:
                        has_min_num_tels = False
                if not has_min_num_tels:
                    pass
                
                # If example_type is a tel_type, add all images
                # from selected telescopes of that type
                if example_type in self.selected_telescopes: 
                    for tel_id in self.selected_telescopes[example_type]:
                        passing_examples.append((row["run_id"], row["event_id"], row["tel_id"]))
                        self.passing_num_examples_by_particle_id[particle_id] += 1
                # if example type is 
                elif example_type == "event":                               
                    passing_examples.append((row["run_id"], row["event_id"]))
                    self.passing_num_examples_by_particle_id[particle_id] += 1

        # get total number of examples
        num_examples = 0
        for particle_id in self.passing_num_examples_by_particle_id:
            num_examples += self.passing_num_examples_by_particle_id[particle_id]

        # compute class weights
        self.class_weights = []
        for particle_id in sorted(self.passing_num_examples_by_particle_id, key=lambda x: self.ids_to_labels[x]):
            class_weights.append(num_examples/float(self.passing_num_examples_by_particle_id[particle_id]))

        # divide passing events into training and validation sets

        # use random seed to get reproducable training
        # and validation sets
        if self.seed is not None:
            random.seed(self.seed)

        random.shuffle(passing_examples)

        if self.mode == 'train':
            # Split examples into training and validation sets
            num_validation = math.ceil(self.validation_split * len(passing_examples)) 
           
            self.training_examples = passing_examples[num_validation:len(passing_examples)]
            self.validation_examples = passing_examples[0:num_validation]

        elif self.mode == 'test':

            self.examples = passing_examples

    # Given a list of examples (tuples), returns a generator function 
    # which yields from the list. Optionally shuffles the examples
    @staticmethod
    def _get_generator_function(examples_list, shuffle=True):

        def generator_fn():
            if shuffle:
                random.shuffle(examples_list)
            for example in examples_list:
                yield example

        return generator_fn

    def get_example_generators(self):

        if self.mode == "train":
            # Convert lists of training and validation examples into generators
            training_generator_fn = self._get_generator_fn(self.training_examples)
            validation_generator_fn = self._get_generator_fn(self.validation_examples)

            return training_generator_fn, validation_generator_fn, self.class_weights
        
        elif self.mode == "test":

            test_generator_fn = self._get_generator_fn(self.examples)

            return test_generator_fn, self.class_weights

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

