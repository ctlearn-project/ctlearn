from operator import itemgetter
import threading
import math
from collections import OrderedDict
import random
from abc import ABC, abstractmethod

import tables
import numpy as np

from ctlearn.data_processing import DataProcessor
from ctlearn.image_mapping import ImageMapper

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
    def add_data_processor(self, data_processor):
        pass

    @abstractmethod
    def get_example_generators(self):
        pass

# PyTables HDF5 implementation of DataLoader
# Corresponds to standard CTA ML format specified by
# ImageExtractor (https://github.com/cta-observatory/image-extractor).
class HDF5DataLoader(DataLoader):

    @staticmethod
    def __synchronized_open_file(*args, **kwargs):
        with threading.Lock() as lock:
            return tables.open_file(*args, **kwargs)

    @staticmethod
    def __synchronized_close_file(self, *args, **kwargs):
        with threading.Lock() as lock:
            return self.close(*args, **kwargs)

    def __init__(self, 
            file_list,
            mode='train',
            example_type='array',
            selected_tel_type='LST',
            selected_tel_ids=None,
            min_num_tels=1,
            cut_condition=None,
            validation_split=0.1,
            use_telescope_positions=True,
            data_processor=None,
            image_mapper=ImageMapper(),
            seed=None
            ):

        # construct dict of filename:file_handle pairs 
        self.files = {filename:self.__synchronized_open_file(filename, mode='r')
                    for filename in file_list}

        # Data loading settings
        if mode in ['train', 'test']:
            self.mode = mode
        else:
            raise ValueError("Invalid mode selection: {}. Select 'train' or 'test'.".format(mode))
        
        if example_type in ['single_tel', 'array']:
            self.example_type = example_type
        else:
            raise ValueError("Invalid example type selection: {}. Select 'single_tel' or 'array'.".format(example_type))

        self.cut_condition = cut_condition
        self.min_num_tels = min_num_tels

        if validation_split < 1.0 and validation_split > 0.0:
            self.validation_split = validation_split
        else:
            raise ValueError("Invalid validation split: {}. Must be between 0.0 and 1.0".format(validation_split))
        
        self.use_telescope_positions = use_telescope_positions

        if isinstance(data_processor, DataProcessor) or data_processor is None:
            self.data_processor = data_processor
        else:
            raise ValueError("data_processor must be an object of type DataProcessor or None.")
        
        self.seed = seed

        # Overwrite self._image_mapper with the ImageMapper of the DataProcessor
        # if one is provided.
        self._image_mapper = image_mapper
        if self.data_processor is not None:
            self._image_mapper = self.data_processor._image_mapper

        # Compute and save metadata describing dataset
        self._load_metadata()
        
        # Select desired telescopes
        self._select_telescopes(selected_tel_type, tel_ids=selected_tel_ids)

        # Apply cuts to get lists of valid examples
        self._apply_cuts()

        # Based on example_type and selected telescopes, compute the generator
        # output datatypes and map_fn output datatypes.
        # NOTE: these dtypes will ultimately be converted to TF datatypes using
        # tf.as_dtype()
        if self.example_type == 'single_tel':
            self.generator_output_dtypes = [np.dtype(np.int64), np.dtype(np.int64), np.dtype(np.int64)] 
            
            data_dtypes = [np.dtype(np.float32)]
            label_dtypes = [np.dtype(np.int64)]

            self.output_names = ['telescope_data', 'gamma_hadron_label']
            self.output_is_label = [False, True]
            self.map_fn_output_dtypes = data_dtypes + label_dtypes
                    
        elif self.example_type == 'array':
            self.generator_output_dtypes = [np.dtype(np.int64), np.dtype(np.int64)] 
           
            data_dtypes = [np.dtype(np.float32), 
                   np.dtype(np.int8),
                   np.dtype(np.float32)
                   ]

            label_dtypes = [np.dtype(np.int64)]
            
            self.output_names = ['telescope_data', 'telescope_triggers', 'telescope_aux_inputs', 'gamma_hadron_label']
            self.output_is_label = [False, False, False, True]
            self.map_fn_output_dtypes = data_dtypes + label_dtypes

    def add_data_processor(self, data_processor):
        if isinstance(data_processor, DataProcessor):
            self.data_processor = data_processor
        else:
            raise ValueError("Must provide a DataProcessor object.")
        self._image_mapper = self.data_processor._image_mapper

    # Compute and save a collection of metadata parameters
    # which describe the dataset
    def _load_metadata(self):

        self.particle_ids = set()

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

        self.__tel_id_to_type_index = {}

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
                self.__tel_id_to_type_index[tel_id] = (tel_type,index)
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

            self.max_telescope_position = [max_tel_x, max_tel_y, max_tel_z]
            
            for row in f.root.Event_Info.iterrows():
                if particle_id not in self.num_events_by_particle_id:
                    self.num_events_by_particle_id[particle_id] = 0

                self.events.append((row['run_number'],row['event_number']))
                self.__events_to_indices[(row['run_number'],row['event_number'])] = (filename, row.nrow)

                self.num_events_by_particle_id[particle_id] += 1
                self.num_events += 1

                for tel_type in self.telescopes:
                    tel_ids = self.telescopes[tel_type]
                    indices = row[tel_type + '_indices']
                    if not tel_type in self.num_images:
                        self.num_images[tel_type] = 0
                    if not tel_type in self.images:
                        self.images[tel_type] = []
                    if not tel_type in self.num_images_by_particle_id:
                        self.num_images_by_particle_id[tel_type] = {}
                    for tel_id, image_index in zip(tel_ids, indices):
                        self.__single_tel_examples_to_indices[(row['run_number'], row['event_number'], tel_id)] = (filename, tel_type, image_index)
                        if image_index != 0:
                            self.images[tel_type].append((row['run_number'], row['event_number'], tel_id))
                            self.num_images[tel_type] += 1
                            if particle_id not in self.num_images_by_particle_id[tel_type]:
                                self.num_images_by_particle_id[tel_type][particle_id] = 0
                            self.num_images_by_particle_id[tel_type][particle_id] += 1

            # Compute max and min pixel value in each telescope image
            # type for normalization
            # NOTE: This step is time-intensive.
            for tel_type in self.telescopes.keys():
                tel_table = f.root._f_get_child(tel_type)
                records = tel_table.read(1,tel_table.shape[0])
                images = records['image_charge']

                if tel_type not in self.image_charge_mins:
                    self.image_charge_mins[tel_type] = np.amin(images)
                if tel_type not in self.image_charge_maxes:
                    self.image_charge_maxes[tel_type] = np.amax(images)

                if np.amin(images) < self.image_charge_mins[tel_type]:
                    self.image_charge_mins[tel_type] = np.amin(images)
                if np.amax(images) > self.image_charge_maxes[tel_type]:
                    self.image_charge_maxes[tel_type] = np.amax(images)
        
            # create mapping from particle ids to labels
            # and from labels to names
            self.ids_to_labels = {particle_id:i 
                    for i, particle_id in enumerate(sorted(list(self.particle_ids)))}
            self.labels_to_names = {i:PARTICLE_ID_TO_NAME[particle_id] 
                    for particle_id, i in self.ids_to_labels.items()}

    # Method returning a dict of selected metadata parameters
    def get_metadata(self):

        metadata = {
                'num_classes': len(list(self.particle_ids)),
                'particle_ids': self.particle_ids,
                'telescopes': self.telescopes,
                'num_telescopes': {self.selected_telescope_type: len(self.selected_telescopes)},
                'selected_telescope_types': [self.selected_telescope_type],
                'num_events_by_particle_id': self.num_events_by_particle_id,
                'num_images_by_particle_id': self.num_images_by_particle_id,
                'num_position_coordinates': self.num_position_coordinates,
                'class_to_name': self.labels_to_names
           }

        if self.data_processor is not None:
            metadata = {**metadata, **self.data_processor.get_metadata()}
        else:
            metadata['num_additional_aux_params'] = 0

        metadata['total_aux_params'] = 0
        if self.use_telescope_positions:
            metadata['total_aux_params'] += 3
        metadata['total_aux_params'] += metadata['num_additional_aux_params']

        return metadata

    # Return dictionary of auxiliary data.
    def get_auxiliary_data(self):
        auxiliary_data = {
            'telescope_positions': self.telescope_positions
            }

        return auxiliary_data

    # Select which telescopes from the full dataset to include in each event 
    # by a telescope type and an optional list of telescope ids.
    def _select_telescopes(self, tel_type, tel_ids=None):
       
        if not tel_ids:
            if tel_type not in self.telescopes:
                raise ValueError("Selected tel type {} not found in dataset.".format(tel_type))
            elif tel_type not in self._image_mapper.mapping_tables:
                raise NotImplementedError("Mapping table for selected tel type {} not implemented.".format(tel_type))
            else:
                self.selected_telescope_type = tel_type
                self.selected_telescopes = self.telescopes[tel_type]
        else:
            self.selected_telescopes = []
            all_tel_ids = {}
            for tel_type in self.telescopes:
                for tel_id in self.telescopes[tel_type]:
                    all_tel_ids[tel_id] = tel_type
            if len(set([all_tel_ids[tel_id] for tel_id in tel_ids])) > 1:
                raise ValueError("Cannot select telescopes of multiple types.")
            for tel_id in tel_ids:
                if all_tel_ids[tel_id] != tel_type:
                    raise ValueError("Selected tel id {} is of wrong tel type {}.".format(tel_id, all_tel_ids[tel_id]))
                elif tel_id not in all_tel_ids:
                    raise ValueError("Selected tel id {} not found in dataset.".format(tel_id))
                elif all_tel_ids[tel_id] not in self._image_mapper.mapping_tables:
                    raise NotImplementedError(
                            "Mapping table for tel type {} of selected tel id {} not implemented.".format(
                                all_tel_ids[tel_id],tel_id))
                else:
                    self.selected_telescopes.append(tel_id)
            self.selected_telescope_type = tel_type
  
    # Get a single telescope image from a particular event, 
    # uniquely identified by a tuple (run_number, event_number, tel_id).
    # The raw 1D trace is transformed into a 1-channel 2D image using a
    # mapping table but no other processing is done.
    def get_image(self, run_number, event_number, tel_id):
        
        tel_type, _ = self.__tel_id_to_type_index[tel_id]
        if tel_type not in self._image_mapper.mapping_tables:
            raise NotImplementedError("Requested image from tel_type {} without valid mapping table.".format(tel_type))

        # get filename, image table name (telescope type), and index
        # corresponding to the desired image
        filename, tel_type, index = self.__single_tel_examples_to_indices[(run_number, event_number, tel_id)]
        
        f = self.files[filename]
        record = f.root._f_get_child(tel_type)[index]
        
        # Allocate empty numpy array of shape (len_trace + 1,) to hold trace plus
        # "empty" pixel at index 0 (used to fill blank areas in image)
        trace = np.empty(shape=(record['image_charge'].shape[0] + 1),dtype=np.float32)
        # Read in the trace from the record 
        trace[0] = 0.0
        trace[1:] = record['image_charge']

        # Create image by indexing into the trace using the mapping table, then adding a
        # dimension to given shape (length,width,1)
        image = self._image_mapper.map_image(trace, tel_type)

        return image

    def get_example(self, *identifiers):

        if self.example_type == "single_tel":

            run_number, event_number, tel_id = identifiers

            # get image from image table
            image = self.get_image(run_number, event_number, tel_id) 

            # locate corresponding event record to get particle type
            filename, index = self.__events_to_indices[(run_number, event_number)]
            f = self.files[filename]
            event_record = f.root.Event_Info[index]

            # Get classification label by converting CORSIKA particle code
            label = self.ids_to_labels[event_record['particle_id']] 
            
            data = [image]
            labels = [label]

        elif self.example_type == "array":

            run_number, event_number = identifiers
            tel_type = self.selected_telescope_type

            # get filename, image table name (telescope type), and index
            # corresponding to the desired image
            filename, index = self.__events_to_indices[(run_number, event_number)]
            
            f = self.files[filename]
            record = f.root.Event_Info[index]

            # Get classification label by converting CORSIKA particle code
            label = self.ids_to_labels[record['particle_id']] 
          
            # Collect images and binary trigger values only for telescopes
            # in selected_telescopes            
            images = []
            triggers = []
            aux_inputs = []
            image_shape = self._image_mapper.image_shapes[tel_type] 
            for tel_id in self.selected_telescopes:
                _, index = self.__tel_id_to_type_index[tel_id]
                i = record[tel_type + "_indices"][index]
                if i == 0:
                    # Telescope did not trigger. Its outputs will be dropped
                    # out, so input is arbitrary. Use an empty array for
                    # efficiency.
                    images.append(np.empty(image_shape))
                    triggers.append(0)  
                else:
                    image = self.get_image(run_number, event_number, tel_id)
                    images.append(image)
                    triggers.append(1)
                if self.use_telescope_positions:
                    telescope_position = self.telescope_positions[tel_type][tel_id]
                    telescope_position = [float(telescope_position[i]) / self.max_telescope_position[i] 
                            for i in range(self.num_position_coordinates)]
                    aux_inputs.append(telescope_position)
            
            data = [images, triggers, aux_inputs]
            labels = [label]

        if self.data_processor:
            data, labels = self.data_processor.process_example(data, labels, self.selected_telescope_type)

        if self.example_type == 'single_tel':
            data[0] = np.stack(data[0]).astype(np.float32)
        elif self.example_type == 'array':
            data[0] = np.stack(data[0]).astype(np.float32)
            data[1] = np.array(data[1], dtype=np.int8)
            data[2] = np.array(data[2], dtype=np.float32)

        return data + labels

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
            rows = [row for row in event_table.where(self.cut_condition)] if self.cut_condition else event_table.iterrows()
            for row in rows:
                # First check if min num tels cut is passed
                if np.count_nonzero(row[self.selected_telescope_type + "_indices"]) < self.min_num_tels:
                    pass
               
                # If example_type is single_tel, there will 
                # be only a single selected telescope type
                if self.example_type == "single_tel":
                    image_indices = row[self.selected_telescope_type + "_indices"]
                    for tel_id in self.selected_telescopes:
                        _, index = self.__tel_id_to_type_index[tel_id]
                        if image_indices[index] != 0:
                            passing_examples.append((row["run_number"], row["event_number"], tel_id))
                            self.passing_num_examples_by_particle_id[particle_id] += 1
                # if example type is 
                elif self.example_type == "array":                               
                    passing_examples.append((row["run_number"], row["event_number"]))
                    self.passing_num_examples_by_particle_id[particle_id] += 1

        # get total number of examples
        num_examples = 0
        for particle_id in self.passing_num_examples_by_particle_id:
            num_examples += self.passing_num_examples_by_particle_id[particle_id]

        # compute class weights
        self.class_weights = []
        for particle_id in sorted(self.passing_num_examples_by_particle_id, key=lambda x: self.ids_to_labels[x]):
            self.class_weights.append(num_examples/float(self.passing_num_examples_by_particle_id[particle_id]))

        # divide passing events into training and validation sets

        # use random seed to get reproducible training
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
            training_generator_fn = self._get_generator_function(self.training_examples)
            validation_generator_fn = self._get_generator_function(self.validation_examples)

            return training_generator_fn, validation_generator_fn, self.class_weights
        
        elif self.mode == "test":

            test_generator_fn = self._get_generator_function(self.examples)

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

