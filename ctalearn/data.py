from operator import itemgetter
import threading

import tables
import numpy as np

IMAGE_SHAPES = {
        'MSTS': (120,120,1)
        }

def __generate_table_MSTS():
    """
    Function returning MSTS mapping table (used to index into the trace when converting from trace to image).
    """
    
    ROWS = 15
    MODULE_DIM = 8
    MODULES_PER_ROW = [
        5,
        9,
        11,
        13,
        13,
        15,
        15,
        15,
        15,
        15,
        13,
        13,
        11,
        9,
        5]
    
    # bottom left corner of each 8 x 8 module in the camera
    # counting from the bottom row, left to right
    MODULE_START_POSITIONS = [(((IMAGE_SHAPES['MSTS'][0] - MODULES_PER_ROW[j] *
                                 MODULE_DIM) / 2) +
                               (MODULE_DIM * i), j * MODULE_DIM)
                              for j in range(ROWS)
                              for i in range(MODULES_PER_ROW[j])]

    table = np.zeros(shape=(IMAGE_SHAPES['MSTS'][0],IMAGE_SHAPES['MSTS'][1]),dtype=int)   
    # Fill appropriate positions with indices
    # NOTE: we append a 0 entry to the (11328,) trace array to allow us to use fancy indexing to fill
    # the empty areas of the (120,120) image. Accordingly, all indices in the mapping table are increased by 1
    # (j starts at 1 rather than 0)
    j = 1
    for (x_0,y_0) in MODULE_START_POSITIONS:
        for i in range(MODULE_DIM * MODULE_DIM):
            x = int(x_0 + i // MODULE_DIM)
            y = y_0 + i % MODULE_DIM
            table[x][y] = j
            j += 1

    return table

MAPPING_TABLES = {
        'MSTS': __generate_table_MSTS()
        }

# Multithread-safe PyTables open and close file fns
lock = threading.Lock()

def synchronized_open_file(*args, **kwargs):
    with lock:
        return tables.open_file(*args, **kwargs)

def synchronized_close_file(self, *args, **kwargs):
    with lock:
        return self.close(*args, **kwargs)

# Generator function used to produce a dataset of elements (HDF5_filename,index)
# from a list of files and a list of lists of indices per file (constructed by applying cuts)
def HDF5_gen_fn(file_list,indices_by_file):
    for i,filename in enumerate(file_list):
        for j in indices_by_file[i]:
            yield (filename.encode('utf-8'),j)

# Data loading function for event-wise (array-level) HDF5 data loading
def load_HDF5_data(filename, index, auxiliary_data, metadata,sort_telescopes_by_trigger=False):

    # Read the event record for the given filename and index
    f = synchronized_open_file(filename.decode('utf-8'), mode='r')
    record = f.root.Event_Info[index]
    
    # Get classification label by converting CORSIKA particle code
    if record['particle_id'] == 0: # gamma ray
        gamma_hadron_label = 1
    elif record['particle_id']  == 101: # proton
        gamma_hadron_label = 0
  
    # Collect image indices for each telescope type in the data
    telescope_types = metadata['telescope_types']
    image_indices = {tel_type:record[tel_type+"_indices"] for tel_type in telescope_types}
   
    # Collect images and binary trigger values
    telescope_images = []
    telescope_triggers = []
    for tel_type in sorted(image_indices):
        # Only save MSTS (other telescope types do not have valid MAPPING_TABLES yet)
        if tel_type in MAPPING_TABLES:
            for i in image_indices[tel_type]:
                if i == 0:
                    # Telescope did not trigger. Its outputs will be dropped
                    # out, so input is arbitrary. Use an empty array for
                    # efficiency.
                    telescope_images.append(np.empty(metadata['image_shapes']['MSTS']))
                    telescope_triggers.append(0)
                else:
                    telescope_image = load_HDF5_image(f,'MSTS',i)
                    telescope_images.append(telescope_image)
                    telescope_triggers.append(1)
    
    synchronized_close_file(f)

    # Collect flattened telescope positions from auxiliary data
    telescope_positions = auxiliary_data

    if sort_telescopes_by_trigger:
        # Group the positions by telescope (also, making a copy prevents
        # modifying the original list)
        group_size = metadata['num_auxiliary_inputs']
        telescope_positions = [telescope_positions[n:n+group_size] for n in
                range(0, len(telescope_positions), group_size)]
        # Sort the images, triggers, and grouped positions by trigger, listing
        # the triggered telescopes first
        telescope_images, telescope_triggers, telescope_positions = map(list,
                zip(*sorted(zip(telescope_images, telescope_triggers,
                    telescope_positions), reverse=True, key=itemgetter(1))))
        # Reflatten the position list
        telescope_positions = [i for sublist in telescope_positions for i in
                sublist]
 
    # Convert to numpy arrays with correct types
    telescope_images = np.stack(telescope_images).astype(np.float32)
    telescope_triggers = np.array(telescope_triggers, dtype=np.int8)
    telescope_positions = np.array(telescope_positions, dtype=np.float32)

    return [telescope_images, telescope_triggers, telescope_positions,
            gamma_hadron_label]

# Data loading function for single tel HDF5 data
# NOTE: Hardcoded to work only for MSTS images
def load_HDF5_data_single_tel(filename, index, metadata):

    # Load image table record from specified file and image table index
    f = synchronized_open_file(filename.decode('utf-8'), mode='r')
    telescope_image = load_HDF5_image(f,'MSTS',index)

    # Get corresponding event record using event_index column
    event_index = f.root._f_get_child('MSTS')[index]['event_index']
    event_record = f.root.Event_Info[event_index]

    # Get classification label by converting CORSIKA particle code
    if event_record['particle_id'] == 0: # gamma ray
        gamma_hadron_label = 1
    elif event_record['particle_id'] == 101: # proton
        gamma_hadron_label = 0
      
    synchronized_close_file(f)

    return [telescope_image, gamma_hadron_label]

# Return dict of auxiliary data values (currently only contains telescope position coordinates).
# Contains all MSTS telescope positions by telescope id.
# Checks that the same telescopes have the same position across all files.
# NOTE: currently hardcoded to only collect MSTS telescope positions (in line with other functions)
def load_HDF5_auxiliary_data(file_list): 
    telescope_positions = {}
    for filename in file_list:
        with tables.open_file(filename, mode='r') as f:
            # For every telescope in the telescope table
            for row in f.root.Telescope_Info.iterrows():
                if row['tel_type'].decode('utf-8') == 'MSTS':
                    if row['tel_id'] not in telescope_positions:
                        telescope_positions[row['tel_id']] = [row["tel_x"],row["tel_y"],row["tel_z"]]
                    else:
                        if telescope_positions[row['tel_id']] != [row["tel_x"],row["tel_y"],row["tel_z"]]:
                            raise ValueError("Telescope positions do not match for telescope {} in file {}.".format(row['tel_id'],filename))
    auxiliary_data = {
            'telescope_positions': telescope_positions
            }
    
    return auxiliary_data

def load_HDF5_metadata(file_list):
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
            # Check that telescope types are the same across all files
            if not telescope_types:
                telescope_types = sorted(list({row["tel_type"].decode('utf-8') for row in f.root.Telescope_Info.iterrows()}))
            else:
                if telescope_types != sorted(list({row["tel_type"].decode('utf-8') for row in f.root.Telescope_Info.iterrows()})):
                    raise ValueError("Telescope types do not match in file {}.".format(filename))
     
            # Check that telescope ids are the same across all files
            if not telescope_ids:
                for tel_type in telescope_types:
                    telescope_ids[tel_type] = sorted([row["tel_id"] for row in f.root.Telescope_Info.iterrows() if row["tel_type"].decode('utf-8') == tel_type])
            else:
                for tel_type in telescope_types:
                    if telescope_ids[tel_type] != sorted([row["tel_id"] for row in f.root.Telescope_Info.iterrows() if row["tel_type"].decode('utf-8') == tel_type]):
                            raise ValueError("Telescope ids do not match in file {}".format(filename))

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
            'num_auxiliary_inputs':3,
            'image_charge_min': image_charge_min,
            'image_charge_max': image_charge_max
            }

    return metadata

def load_HDF5_image(data_file,tel_type,index):
    record = data_file.root._f_get_child(tel_type)[index]
    telescope_image = []
    
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
def apply_cuts_HDF5(data_files,cut_condition,model_type):
    indices_by_file = []
    for filename in data_files:
        # No need to use the multithread-safe file open, as this function is only called once
        with tables.open_file(filename, mode='r') as f:
            # For single tel, get all passing events, then collect all non-zero MSTS image indices into a flat list
            if model_type == 'singletel':
                passing_events = f.root.Event_Info.where(cut_condition) if cut_condition is not None else f.root.Event_Info.iterrows()
                indices = [i for row in passing_events for i in row['MSTS_indices'] if i != 0]
            # For array-level get all passing rows and return a list of all of the indices
            else:
                indices = [row.nrow for row in table.where(cut_condition)] if cut_condition is not None else range(table.nrows)
            
            indices_by_file.append(indices)

    return indices_by_file

