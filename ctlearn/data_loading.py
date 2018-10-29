from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
import math
import random
import threading

import numpy as np
import tables

from ctlearn.data_processing import DataProcessor
from ctlearn.image_mapping import ImageMapper

# Maps CORSIKA particle id codes
# to particle class names
PARTICLE_ID_TO_CLASS_NAME = {
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
            use_peak_times=False,
            example_type='array',
            selected_tel_types=None,
            selected_tel_ids=None,
            merge_tel_types=False,
            min_num_tels=1,
            cut_condition=None,
            validation_split=0.1,
            use_telescope_positions=True,
            data_processor=None,
            image_mapper=None,
            seed=None
            ):

        # construct dict of filename:file_handle pairs 
        self.files = OrderedDict()
        for filename in file_list:
            self.files[filename] = \
                self.__synchronized_open_file(filename, mode='r')

        # Data loading settings
        if mode in ['train', 'test']:
            self.mode = mode
        else:
            raise ValueError("Invalid mode selection: {}. Select 'train' or 'test'.".format(mode))
        
        self.use_peak_times = use_peak_times
        
        if example_type in ['single_tel', 'array']:
            self.example_type = example_type
        else:
            raise ValueError("Invalid example type selection: {}. Select 'single_tel' or 'array'.".format(example_type))

        if selected_tel_types is None:
            selected_tel_types = ['LST']
        self.selected_telescope_types = selected_tel_types
        if not self.selected_telescope_types:
            raise ValueError("No telescope types selected.")
            
        if (self.example_type == 'single_tel'
                and len(self.selected_telescope_types) != 1):
            raise ValueError("Must select exactly one tel type in single tel "
                    "mode. Selected: {}".format(self.selected_telescope_types))

        if selected_tel_ids is None:
            selected_tel_ids = {}
        self.selected_tel_ids = selected_tel_ids
        self.merge_tel_types = merge_tel_types
       
        self.min_num_tels = min_num_tels
        self.cut_condition = cut_condition

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
        if self.data_processor is not None:
            self._image_mapper = self.data_processor._image_mapper
        elif image_mapper is not None:
            self._image_mapper = image_mapper
        else:
            self._image_mapper = ImageMapper(use_peak_times=self.use_peak_times)

        # Compute and save metadata describing dataset
        self._load_metadata()
        
        # If needed for normalization, update min max charge values
        # And feed them to the data processor
        if self.data_processor is not None and \
            self.data_processor.normalization is not None:
            
            self.image_charge_mins = {}
            self.image_charge_maxes = {}
                
            for filename in self.files:
                self._update_min_max_charge_values(filename)
            
            self.data_processor.add_image_charge_mins(
                    self.image_charge_mins
                    )
        
        # Select desired telescopes
        self._select_telescopes()

        # Apply cuts to get lists of valid examples
        self._apply_cuts()

        # Compute class weights to normalize uneven classes
        self._compute_class_weights()

        # Split into training/validation/test sets
        self._split_into_datasets()

        # Based on example_type and selected telescopes, compute the generator
        # output datatypes and map_fn output names and datatypes.
        # NOTE: these dtypes will ultimately be converted to TF datatypes using
        # tf.as_dtype()
        if self.example_type == 'single_tel':
            generator_outputs = {
                    'run_number': np.dtype(np.int64),
                    'event_number': np.dtype(np.int64),
                    'tel_id': np.dtype(np.int64)
                    }
            data_outputs = {
                    # camera images
                    'data': np.dtype(np.float32)
                    }
            tel_types = ['telescope']
        elif self.example_type == 'array':
            generator_outputs = {
                    'run_number': np.dtype(np.int64),
                    'event_number': np.dtype(np.int64)
                    }
            data_outputs = {
                    # camera images
                    'data': np.dtype(np.float32),
                    # binary trigger values (0 or 1)
                    'triggers': np.dtype(np.int8),
                    # position coordinates and other info
                    'aux_inputs': np.dtype(np.float32)
                    }
            tel_types = (['telescope'] if self.merge_tel_types
                    else self.selected_telescope_types)

        label_outputs = {
                # gamma / proton classification label
                'gamma_hadron_label': np.dtype(np.int64)
                }

        self.generator_output_dtypes = [dtype for dtype in
                generator_outputs.values()] 

        self.output_names = []
        self.output_is_label = []
        self.map_fn_output_dtypes = []
        
        for tel_type in tel_types:
            for name, dtype in data_outputs.items():
                self.output_names.append(tel_type + '_' + name)
                self.output_is_label.append(False)
                self.map_fn_output_dtypes.append(dtype)
        
        for name, dtype in label_outputs.items():
            self.output_names.append(name)
            self.output_is_label.append(True)
            self.map_fn_output_dtypes.append(dtype)

    # Compute and save a collection of metadata parameters
    # which describe the dataset
    def _load_metadata(self):
        
        # INITIALIZE ALL METADATA VARIABLES 
        
        self.class_names = set()

        # OrderedDict with telescope types as keys and list of telescope ids
        # of each type (in sorted order) as values
        # NOTE: the telescope types are ordered by increasing telescope id
        self.total_telescopes = OrderedDict()

        self.events = [] 
        self.images = {}

        self.num_events = 0

        self.num_events_before_cuts_by_class_name = {}
        self.num_images_before_cuts_by_tel_and_class_name = {}

        self.num_position_coordinates = 3
        self.telescope_positions = {}
        self.max_telescope_positions = {}
       
        self.__events_to_indices = {}
        self.__single_tel_examples_to_indices = {}
        self.__tel_id_to_tel_type = {}
        
        # PROCESS METADATA ACROSS ALL DATA FILES

        for filename in self.files:
            
            # Process telescope array metadata
            self._process_array_info(filename)
            
            # Process event info
            self._process_events(filename)
        
        # Updates max telescope coordinates
        self._update_max_coordinates()
        
        # create mapping from particle ids to labels
        # and from labels to names
        self.class_names_to_labels = {class_name:i 
                for i, class_name in enumerate(sorted(list(self.class_names)))}
        self.labels_to_class_names = {i:class_name 
                for i, class_name in enumerate(sorted(list(self.class_names)))}
    
    # Creates a sorted dict that associates telescope_types with the list
    # of telescope ids corresponding to that type of telescope
    # Also updates telescopes position and checks that those are consistent
    # across files
    def _process_array_info(self, filename):
        # get file handle
        f = self.files[filename]
        
        telescopes = {}
        for row in f.root.Array_Info.iterrows():
            # note: tel type strings stored in Pytables as byte strings, must be decoded
            tel_type = row['tel_type'].decode('utf-8')
            tel_id = row['tel_id']
            
            # Store the telescope for indexing
            if tel_type not in telescopes:
                telescopes[tel_type] = []
                
            # There is an overflow issue with the tel_id parameter in
            # the ImageExtractor v0.5.1 data format. The tel_ids > 255 
            # have looped back around to 0. SST1 is the only affected 
            # telescope type. SST1 has tel_id 220-255 followed by "0-33".
            if tel_type == 'SST1' and tel_id < 220:
                tel_id += 256
                
            telescopes[tel_type].append(tel_id)
            self.__tel_id_to_tel_type[tel_id] = tel_type
            
            # Store the telescope position
            if tel_type not in self.telescope_positions:
                self.telescope_positions[tel_type] = {}
            if tel_id not in self.telescope_positions[tel_type]:
                self.telescope_positions[tel_type][tel_id] = \
                        [row["tel_x"], row["tel_y"], row["tel_z"]]
            else:
                if self.telescope_positions[tel_type][tel_id] != \
                        [row["tel_x"], row["tel_y"], row["tel_z"]]:
                    raise ValueError("Telescope positions do not match for telescope {} in file {}.".format(tel_id,filename))

        # Sort the telescopes by tel type and id
        telescopes = OrderedDict(sorted(telescopes.items(),
            key=lambda i: i[0]))
        for tel_type in telescopes:
            telescopes[tel_type].sort()

        if not self.total_telescopes:
            self.total_telescopes = telescopes
        else:
            # Check if the telescope array is the same across files
            if self.total_telescopes != telescopes:
                raise ValueError("Telescope type/id mismatch in file {}".format(filename))
    
    # Updates the max telescope coordinates seen so far for normalization
    def _update_max_coordinates(self):
        # get a file handle to extract the data
        # Any file will do since all files should have the same array data
        filename = next(iter(self.files))
        f = self.files[filename]
        
        # Compute max x, y, z telescope coordinates for normalization
        max_tel_x = max(row['tel_x'] for row in f.root.Array_Info.iterrows())
        max_tel_y = max(row['tel_y'] for row in f.root.Array_Info.iterrows())
        max_tel_z = max(row['tel_z'] for row in f.root.Array_Info.iterrows())

        self.max_telescope_position = [max_tel_x, max_tel_y, max_tel_z]
        
    # Stores the position of each image in the file, indexed by telescope type
    # Also keeps track of the classes and number of events, total and per class
    def _process_events(self, filename):
        # get file handle
        f = self.files[filename]
        
        # Particle ID is same for all events in a given file and
        # is therefore saved in the root attributes
        class_name = PARTICLE_ID_TO_CLASS_NAME[f.root._v_attrs.particle_type]
        self.class_names.add(class_name)
        
        # If no previous events of this class had been loaded before,
        # we start the counter for them
        if class_name not in self.num_events_before_cuts_by_class_name:
            self.num_events_before_cuts_by_class_name[class_name] = 0
        
        # Each row in the file is an event
        for row in f.root.Event_Info.iterrows():
    
            self.events.append((row['run_number'],row['event_number']))
            self.__events_to_indices[(row['run_number'],row['event_number'])] = \
                    (filename, row.nrow)
    
            self.num_events_before_cuts_by_class_name[class_name] += 1
            self.num_events += 1
    
            for tel_type in self.total_telescopes:
                tel_ids = self.total_telescopes[tel_type]
                indices = row[tel_type + '_indices']
                if not tel_type in self.images:
                    self.images[tel_type] = []
                if not tel_type in self.num_images_before_cuts_by_tel_and_class_name:
                    self.num_images_before_cuts_by_tel_and_class_name[tel_type] = {}
                
                # for each image index associated to this event
                for tel_id, image_index in zip(tel_ids, indices):
                    self.__single_tel_examples_to_indices[
                            (row['run_number'], row['event_number'], tel_id)
                            ] = (filename, tel_type, image_index)
                    if image_index != 0:
                        self.images[tel_type].append((row['run_number'], row['event_number'], tel_id))
                        if class_name not in self.num_images_before_cuts_by_tel_and_class_name[tel_type]:
                            self.num_images_before_cuts_by_tel_and_class_name[tel_type][class_name] = 0
                        self.num_images_before_cuts_by_tel_and_class_name[tel_type][class_name] += 1
    
    def _update_min_max_charge_values(self, filename):
        # get file handle
        f = self.files[filename]
        
        # Compute max and min pixel value in each telescope image
        # type for normalization
        # NOTE: This step is time-intensive.
        for tel_type in self.total_telescopes.keys():
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
    
    # Method returning a dict of selected metadata parameters
    def get_metadata(self):

        metadata = {
                'num_classes': len(list(self.class_names)),
                'class_names': self.class_names,
                'total_telescopes': self.total_telescopes,
                'num_total_telescopes': {tel_type: len(tel_ids) for
                    tel_type, tel_ids in self.total_telescopes.items()},
                'selected_telescope_types': self.selected_telescope_types,
                'selected_telescopes': self.selected_telescopes,
                'num_selected_telescopes': {tel_type: len(tel_ids) for
                    tel_type, tel_ids in self.selected_telescopes.items()},
                'num_events_before_cuts_by_class_name': self.num_events_before_cuts_by_class_name,
                'num_images_before_cuts_by_tel_and_class_name': self.num_images_before_cuts_by_tel_and_class_name,
                'num_events_after_cuts_by_class_name': self.num_passing_events_by_class_name,
                'num_images_after_cuts_by_tel_and_class_name': self.num_passing_images_by_tel_and_class_name,
                'num_val_examples_by_class_name': self.num_val_examples_by_class_name,
                'num_position_coordinates': self.num_position_coordinates,
                'labels_to_class_names': self.labels_to_class_names,
                'class_names_to_labels': self.class_names_to_labels
           }

        if self.data_processor is not None:
            metadata = {**metadata, **self.data_processor.get_metadata()}
        else:
            metadata['num_additional_aux_params'] = 0
            metadata['image_shapes'] = self._image_mapper.image_shapes

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
    def _select_telescopes(self):

        self.selected_telescopes = {}
        for tel_type in self.selected_telescope_types:
            # Check that the tel_type is in the data and mapping tables
            if tel_type not in self.total_telescopes:
                raise ValueError("Selected tel type {} not found in "
                        "dataset.".format(tel_type))
            if tel_type not in self._image_mapper.mapping_tables:
                raise NotImplementedError("Mapping table for selected "
                        "tel type {} not implemented.".format(tel_type))
            available_tel_ids = self.total_telescopes[tel_type]
            # Keep only the selected tel ids for the tel type
            if tel_type in self.selected_tel_ids:
                # Check that all requested telescopes are available to select
                requested_tel_ids = self.selected_tel_ids[tel_type]
                invalid_tel_ids = list(set(requested_tel_ids) -
                        set(available_tel_ids))
                if invalid_tel_ids:
                    raise ValueError("Tel ids {} are not a valid selection"
                            "for tel type '{}'".format(invalid_tel_ids,
                                tel_type))
                self.selected_telescopes[tel_type] = requested_tel_ids
            else:
                self.selected_telescopes[tel_type] = available_tel_ids
  
    # Get a single telescope image from a particular event, 
    # uniquely identified by a tuple (run_number, event_number, tel_id).
    # The raw 1D trace is transformed into a 1-channel 2D image using a
    # mapping table but no other processing is done.
    def get_image(self, run_number, event_number, tel_id):
        
        tel_type = self.__tel_id_to_tel_type[tel_id]
        
        if tel_type not in self._image_mapper.mapping_tables:
            raise NotImplementedError("Requested image from tel_type {} without valid mapping table.".format(tel_type))

        # get filename, image table name (telescope type), and index
        # corresponding to the desired image
        filename, tel_type, index = self.__single_tel_examples_to_indices[(run_number, event_number, tel_id)]
        
        f = self.files[filename]
        record = f.root._f_get_child(tel_type)[index]
        
        # Allocate empty numpy array of shape (len_trace + 1,) to hold trace plus
        # "empty" pixel at index 0 (used to fill blank areas in image)
        if self.use_peak_times:
            trace = np.empty(shape=(record['image_charge'].shape[0] + 1, 2),dtype=np.float32)
        else:
            trace = np.empty(shape=(record['image_charge'].shape[0] + 1, 1),dtype=np.float32)
        # Read in the trace from the record 
        trace[0, :] = 0.0
        trace[1:, 0] = record['image_charge']
        
        if self.use_peak_times:
            trace[1:, 1] = record['image_peak_times']

        # Create image by indexing into the trace using the mapping table, then adding a
        # dimension to given shape (length,width,1)
        image = self._image_mapper.map_image(trace, tel_type)
        image = np.array(image, dtype=np.float32)

        return image

    def get_example(self, *identifiers):

        # Get record for the event
        run_number, event_number = identifiers[0:2]
        filename, index = self.__events_to_indices[(run_number, event_number)]
        f = self.files[filename]
        record = f.root.Event_Info[index]
        
        # Get classification label by converting CORSIKA particle code
        class_name = PARTICLE_ID_TO_CLASS_NAME[record['particle_id']]
        labels = [self.class_names_to_labels[class_name]]

        # Get data
        if self.example_type == "single_tel":
            # Easy for single tel - just the image
            tel_id = identifiers[2]
            data = self.get_image(run_number, event_number, tel_id)
            # Add telescope dimension for compatibility with array models
            # when using single tel for generating pretrained weights 
            data = [np.expand_dims(data, axis=0).astype(np.float32)]
        elif self.example_type == "array":
            # Loop over the selected telescopes in the event to get the
            # data, in the form:
            # list over tel_types:
            #   list:
            #       list of images (NumPy arrays),
            #       NumPy array of binary trigger values,
            #       list of auxiliary inputs (NumPy arrays)
            data = []
            for tel_type, tel_ids in self.selected_telescopes.items():
                data.append([[], [], []])
                for tel_id in tel_ids:
                    tel_index = self.total_telescopes[tel_type].index(tel_id)
                    image_index = record[tel_type + "_indices"][tel_index]
                    if image_index == 0:
                        # Telescope didn't trigger, load dummy data
                        image_shape = self._image_mapper.image_shapes[tel_type] 
                        image = np.zeros(image_shape, dtype=np.float32)
                        trigger = 0
                    else:
                        image = self.get_image(run_number, event_number, tel_id)
                        trigger = 1
                    data[-1][0].append(image)
                    data[-1][1].append(np.array(trigger, dtype=np.int8))
                    aux_inputs = []
                    if self.use_telescope_positions:
                        normalized_position = [float(tel_coord) / max_coord
                                for tel_coord, max_coord in zip(
                                    self.telescope_positions[tel_type][tel_id],
                                    self.max_telescope_position)]
                        aux_inputs.append(normalized_position)
                    aux_inputs = np.array(aux_inputs, dtype=np.float32)
                    data[-1][2].append(aux_inputs)

        # Process the example
        if self.data_processor:
            data, labels = self.data_processor.process_example(data, labels,
                    self.selected_telescope_types,
                    example_type=self.example_type)

        if self.example_type == 'array':
            # Combine the data arrays for all telescopes of each type
            for type_i in range(len(data)):
                data[type_i][0] = np.stack(data[type_i][0])
                data[type_i][1] = np.stack(data[type_i][1])
                data[type_i][2] = np.stack(data[type_i][2])
            # If specified, combine the arrays for all tel types
            if self.merge_tel_types:
                data = [np.concatenate(d) for d in zip(*data)]

        return data + labels

    # Get all indices in each HDF5 file which pass a provided cut condition.
    # In single_tel mode, return all image table indices from events
    # passing the cuts. In array-level mode, return all event table indices
    # from events passing the cuts.
    # Apply two kinds of cuts. First, apply cuts specified in the cut
    # condition, a string formatted as a PyTables selection condition (i.e. 
    # for table.where()). See Pytables documentation for examples.
    # If the cut condition is empty, don't apply any cuts here.
    # Second, cut events with fewer than the minimum number of telescopes 
    # as specifed by min_num_tels.
    def _apply_cuts(self):

        self.passing_examples = []
        self.num_passing_events = 0
        self.num_passing_events_by_class_name = {}
        self.num_passing_images = 0
        self.num_passing_images_by_tel_and_class_name = {tel_type: {}
                for tel_type in self.selected_telescopes}
        self.num_passing_examples = 0
        self.num_passing_examples_by_class_name = 0

        for filename in self.files:
            f = self.files[filename]
            particle_id = f.root._v_attrs.particle_type
            class_name = PARTICLE_ID_TO_CLASS_NAME[particle_id]
            if class_name not in self.num_passing_events_by_class_name:
                self.num_passing_events_by_class_name[class_name] = 0
            
            for tel_type in self.num_passing_images_by_tel_and_class_name:
                if class_name not in self.num_passing_images_by_tel_and_class_name[tel_type]:
                    self.num_passing_images_by_tel_and_class_name[tel_type][class_name] = 0
            
            event_table = f.root.Event_Info
            
            # Choose only rows passing the cuts specified in cut condition
            rows = (event_table.where(self.cut_condition) 
                    if self.cut_condition else event_table.iterrows())
            for row in rows:
                # Check that the event has the mininum number of telescopes,
                # ignoring non-selected telescopes
                num_triggered_tels = 0
                num_triggered_tels_by_type = {}
                triggered_tel_ids = []
                for tel_type, tel_ids in self.selected_telescopes.items():
                    tel_id_to_index = {tel_id:
                            self.total_telescopes[tel_type].index(tel_id)
                            for tel_id in tel_ids}
                    tel_indices = np.array(list(tel_id_to_index.values()))
                    image_indices = row[tel_type + "_indices"]
                    triggered_image_indices = image_indices[tel_indices]
                    num_tels = np.count_nonzero(triggered_image_indices)
                    num_triggered_tels += num_tels
                    num_triggered_tels_by_type[tel_type] = num_tels
                    triggered_tel_ids.extend([tel_id for tel_id in tel_ids
                        if tel_id_to_index[tel_id] in triggered_image_indices])
                if num_triggered_tels < self.min_num_tels:
                    continue

                # The event passed all cuts
                self.num_passing_events += 1
                self.num_passing_events_by_class_name[class_name] += 1
                self.num_passing_images += num_triggered_tels
                for tel_type in self.selected_telescopes:
                    self.num_passing_images_by_tel_and_class_name[tel_type][class_name] += num_triggered_tels_by_type[tel_type]

                # Save the passing example(s) depending on the example type
                if self.example_type == 'single_tel':
                    for tel_id in triggered_tel_ids:
                        self.passing_examples.append((row['run_number'],
                            row['event_number'], tel_id))
                elif self.example_type == 'array':
                    self.passing_examples.append((row['run_number'],
                        row['event_number']))
                    
        # Record total number of examples
        if self.example_type == 'single_tel':
            self.num_passing_examples = self.num_passing_images
            self.num_passing_examples_by_class_name = \
                    self.num_passing_images_by_tel_and_class_name[
                            self.selected_telescope_types[0]]
        elif self.example_type == 'array':
            self.num_passing_examples = self.num_passing_events
            self.num_passing_examples_by_class_name = \
                    self.num_passing_events_by_class_name

    def _compute_class_weights(self):
        self.class_weights = []
        for class_name in sorted(self.num_passing_examples_by_class_name,
                key=lambda x: self.class_names_to_labels[x]):
            class_examples = self.num_passing_examples_by_class_name[class_name]
            try:
                class_inverse_frac = self.num_passing_examples / class_examples
            except ZeroDivisionError:
                print("Warning: class '{}' has no examples, unable to "
                        "calculate class weights".format(class_name))
                self.class_weights = [1.0 for c in class_names]
                return
            self.class_weights.append(class_inverse_frac)

    def _split_into_datasets(self):
        
        self.num_val_examples_by_class_name = {}
        
        if self.mode == 'train':            
            # use random seed to get reproducible training
            # and validation sets
            if self.seed is not None:
                random.seed(self.seed)
            
            random.shuffle(self.passing_examples)

            # Split examples into training and validation sets
            num_validation = math.ceil(self.validation_split * len(self.passing_examples)) 
           
            self.training_examples = self.passing_examples[num_validation:len(self.passing_examples)]
            self.validation_examples = self.passing_examples[0:num_validation]
            
            # Count validation examples
            for example in self.validation_examples:
                if self.example_type == 'single_tel':
                    run_number, event_number, _ = example
                elif self.example_type == 'array':
                    run_number, event_number = example
    
                # locate corresponding event record to get particle type
                filename, index = self.__events_to_indices[(run_number, event_number)]
                f = self.files[filename]
                event_record = f.root.Event_Info[index]
    
                # Get classification label by converting CORSIKA particle code
                class_name = PARTICLE_ID_TO_CLASS_NAME[event_record['particle_id']]
                
                if class_name not in self.num_val_examples_by_class_name:
                    self.num_val_examples_by_class_name[class_name] = 0
                
                self.num_val_examples_by_class_name[class_name] += 1

        elif self.mode == 'test':

            self.examples = self.passing_examples

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
            validation_generator_fn = self._get_generator_function(
                    self.validation_examples, shuffle=False)

            return training_generator_fn, validation_generator_fn, self.class_weights
        
        elif self.mode == "test":

            test_generator_fn = self._get_generator_function(self.examples,
                    shuffle=False)

            return test_generator_fn, self.class_weights

    # Log the proportions of classes in the dataset
    def log_class_breakdown(self, logger=None):
    
        if not logger: logger = logging.getLogger()
        
        logger.info("Number of examples after the cuts: {}".format(self.num_passing_examples))
        for class_name in self.num_passing_examples_by_class_name:
            percentage = (100. * self.num_passing_examples_by_class_name[class_name] /
                    self.num_passing_examples)
            logger.info("Number of {} (class {}) examples after the cuts: {} ({:.3f}%)".format(
                class_name,
                self.class_names_to_labels[class_name],
                self.num_passing_examples_by_class_name[class_name],
                percentage))
