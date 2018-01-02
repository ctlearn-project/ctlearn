from operator import itemgetter
import threading

import tables
import numpy as np

IMAGE_SHAPES = {
        'MSTS': (120,120,1)
        }

def __generate_table_MSTS():
    """
    Function returning MSTS injunction table
    """
    
    ROWS = 15
    MODULE_DIM = 8
    MODULE_SIZE = MODULE_DIM * MODULE_DIM
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

    # counting from the bottom row, left to right
    MODULE_START_POSITIONS = [(((IMAGE_SHAPES['MSTS'][0] - MODULES_PER_ROW[j] *
                                 MODULE_DIM) / 2) +
                               (MODULE_DIM * i), j * MODULE_DIM)
                              for j in range(ROWS)
                              for i in range(MODULES_PER_ROW[j])]

    injunction_table = [(int(x_0 + int(i / MODULE_DIM)),y_0 + i % MODULE_DIM) 
                        for (x_0,y_0) in MODULE_START_POSITIONS
                        for i in range(MODULE_SIZE)]

    return injunction_table


INJUNCTION_TABLES = {
        'MSTS': __generate_table_MSTS()
        }


# Functions for locking when opening and closing the same file in multiple threads

lock = threading.Lock()

def synchronized_open_file(*args, **kwargs):
    with lock:
        return tables.open_file(*args, **kwargs)

def synchronized_close_file(self, *args, **kwargs):
    with lock:
        return self.close(*args, **kwargs)

# Generator function used to produce a dataset of elements (HDF5_filename,index)

def HDF5_gen_fn(file_list,num_events,skip_zero=False):
    for i,filename in enumerate(file_list):
        for j in range(num_events[i]):
            if not skip_zero or j != 0:
                yield (filename.encode('utf-8'),j)

# Data loading function for event-wise (array-level) HDF5 data loading
def load_HDF5_data(filename, index, auxiliary_data, metadata,sort_telescopes_by_trigger=False):

    # Read the data at the given table and index from the file
    f = synchronized_open_file(filename.decode('utf-8'), mode='r')
    record = f.root.Event_Info.read(index, index + 1)
    
    # Get classification label by converting CORSIKA particle code
    particle_id = record['particle_id'][0]
    if particle_id == 0: # gamma ray
        gamma_hadron_label = 1
    elif particle_id == 101: # proton
        gamma_hadron_label = 0
   
    telescope_types = metadata['telescope_types']
    image_indices = {tel_type:record[tel_type+"_indices"] for tel_type in telescope_types}
    # list of telescope images
    telescope_images = []
    # list of binary telescope triggers
    telescope_triggers = []
    for tel_type in sorted(image_indices):
        if tel_type in INJUNCTION_TABLES:
            indices = image_indices[tel_type][0]
            for i in indices:
                if i == 0:
                    # Telescope did not trigger. Its outputs will be dropped
                    # out, so input is arbitrary. Use an empty array for
                    # efficiency.
                    telescope_images.append(np.empty(metadata['image_shapes']['MSTS']))
                    telescope_triggers.append(0)
                else:
                    telescope_table = f.root._f_get_child(tel_type)
                    image = np.empty(metadata['image_shapes']['MSTS'])
                    for j,value in enumerate(telescope_table.read(i, i + 1)['image_charge'][0]):
                        x,y = INJUNCTION_TABLES['MSTS'][j]
                        image[x][y][0] = value
                    telescope_images.append(image)
                    telescope_triggers.append(1)
    
    synchronized_close_file(f)

    # telescope positions 
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
 
    # Convert to numpy arrays
    telescope_images = np.stack(telescope_images).astype(np.float32)
    telescope_triggers = np.array(telescope_triggers, dtype=np.int8)
    telescope_positions = np.array(telescope_positions, dtype=np.float32)

    return [telescope_images, telescope_triggers, telescope_positions,
            gamma_hadron_label]

# Data loading function for single tel HDF5 data loading
def load_HDF5_data_single_tel(filename, index, metadata):

    # Read the data at the given table and index from the file
    f = synchronized_open_file(filename.decode('utf-8'), mode='r')
    record = f.root.MSTS.read(index, index + 1)

    # Get image
    telescope_image = np.empty(metadata['image_shapes']['MSTS'],dtype=np.float32)
    for j,value in enumerate(record['image_charge'][0]):
        x,y = INJUNCTION_TABLES['MSTS'][j]
        telescope_image[x][y][0] = value

    event_index = record['event_index'][0] 
    event_record = f.root.Event_Info.read(event_index,event_index+1)

    # Get classification label by converting CORSIKA particle code
    particle_id = event_record['particle_id'][0]
    if particle_id == 0: # gamma ray
        gamma_hadron_label = 1
    elif particle_id == 101: # proton
        gamma_hadron_label = 0
      
    synchronized_close_file(f)

    return [telescope_image, gamma_hadron_label]

def load_HDF5_auxiliary_data(file_list):
  
    telescope_positions = {}
    for filename in file_list:
        with tables.open_file(filename, mode='r') as f:
            for row in f.root.Telescope_Info.iterrows():
                if row["tel_type"].decode('utf-8') == 'MSTS':
                    tel_id = row["tel_id"]
                    position = [row["tel_x"],row["tel_y"],row["tel_z"]]
                    if tel_id not in telescope_positions:
                        telescope_positions[tel_id] = position
                    else:
                        if telescope_positions[tel_id] != position:
                            raise ValueError("Telescope positions do not match for telescope {} in file {}.".format(tel_id,filename))
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

        metadata = {
            'num_events_by_file': num_events_by_file,
            'num_telescopes': {tel_type:len(telescope_ids[tel_type]) for tel_type in telescope_types},
            'telescope_ids': telescope_ids,
            'telescope_types': telescope_types,
            'num_images_by_file': num_images_by_file,
            'particle_id_by_file': particle_id_by_file,
            'image_shapes': IMAGE_SHAPES,
            'num_classes': len(set(particle_id_by_file)),
            'num_auxiliary_inputs':3
            }

    return metadata
