import tables
import numpy as np


def load_HDF5_data_by_tel(filename, index, metadata, mode='TRAIN'):

    # Read the data at the given table and index from the file
    f = tables.open_file(filename, mode='r')
    if mode == 'TRAIN':
        table = f.root.E0.Events_Training
    elif mode == 'VALID':
        table = f.root.E0.Events_Validation
    else:
        raise ValueError("Mode must be 'TRAIN' or 'VALID'")
    record = table.read(index, index + 1)

    tel_data_table = f.root.Tel_Table

    telescope_ids = metadata['telescope_ids']
    image_indices = record['tel_map'][0]
    telescope_images = []
    telescope_positions = []
    for telescope_id, image_index in zip(telescope_ids, image_indices):
       if image_index != -1:
            telescope_table = f.root.E0._f_get_child(telescope_id)
            telescope_images.append(telescope_table[image_index])
            telescope_position = [[x['tel_x'],x['tel_y'],x['tel_z']] for x in tel_data_table.iterrows() if x['tel_id'] == int(telescope_id[1:])][0]
            telescope_positions.append(telescope_position)

    while len(telescope_images) < len(telescope_ids):
        telescope_images.append(np.zeros(metadata['image_shape']))
        telescope_positions.append([0,0,0])
 
    telescope_positions = np.stack(telescope_positions).astype(np.float32)
    telescope_images = np.stack(telescope_images).astype(np.float32)

    # Get binary values indicating whether each telescope triggered
    telescope_triggers = np.array([0 if i < 0 else 1 for i in image_indices].sort(reverse=True),dtype=np.int8)
    
    # Get classification label by converting CORSIKA particle code
    gamma_hadron_label = record['gamma_hadron_label'][0]
    if gamma_hadron_label == 0: # gamma ray
        gamma_hadron_label = 1
    elif gamma_hadron_label == 101: # proton
        gamma_hadron_label = 0
    
    f.close()
    
    return [telescope_images, telescope_triggers, telescope_positions, gamma_hadron_label]

def load_HDF5_data(filename, index, metadata, mode='TRAIN'):

    # Read the data at the given table and index from the file
    f = tables.open_file(filename, mode='r')
    if mode == 'TRAIN':
        table = f.root.E0.Events_Training
    elif mode == 'VALID':
        table = f.root.E0.Events_Validation
    else:
        raise ValueError("Mode must be 'TRAIN' or 'VALID'")
    record = table.read(index, index + 1)
    
    telescope_ids = metadata['telescope_ids']
    image_indices = record['tel_map'][0]
    telescope_images = []
    for telescope_id, image_index in zip(telescope_ids, image_indices):
        if image_index == -1:
            # Telescope did not trigger. Its outputs will be dropped
            # out, so input is arbitrary. Use an empty array for
            # efficiency.
            telescope_images.append(np.empty(metadata['image_shape']))
        else:
            telescope_table = f.root.E0._f_get_child(telescope_id)
            telescope_images.append(telescope_table[image_index])
    telescope_images = np.stack(telescope_images).astype(np.float32)
    
    # Get binary values indicating whether each telescope triggered
    telescope_triggers = np.array([0 if i < 0 else 1 for i in image_indices],
            dtype=np.int8)
    
    # Get classification label by converting CORSIKA particle code
    gamma_hadron_label = record['gamma_hadron_label'][0]
    if gamma_hadron_label == 0: # gamma ray
        gamma_hadron_label = 1
    elif gamma_hadron_label == 101: # proton
        gamma_hadron_label = 0
    
    f.close()
    
    return [telescope_images, telescope_triggers, gamma_hadron_label]

def load_HDF5_auxiliary_data(filename):
    
    f = tables.open_file(filename, mode='r')
    telescope_positions = []
    for row in f.root.Tel_Table.iterrows():
        telescope_positions.append(row["tel_x"])
        telescope_positions.append(row["tel_y"])
        telescope_positions.append(row["tel_z"])
    f.close()
    auxiliary_data = {
        'telescope_positions': np.array(telescope_positions, dtype=np.float32)
        }
    return auxiliary_data

def load_HDF5_metadata(filename):
   
    f = tables.open_file(filename, mode='r')
    num_training_events = f.root.E0.Events_Training.shape[0]
    num_validation_events = f.root.E0.Events_Validation.shape[0]
    # List of telescope IDs ordered by mapping index
    telescope_ids = ["T" + str(row["tel_id"]) for row 
            in f.root.Tel_Table.iterrows()]
    num_telescopes = f.root.Tel_Table.shape[0]
    # All telescope images have the same shape
    image_shape = f.root.E0._f_get_child(telescope_ids[0]).shape[1:]
    f.close()
    metadata = {
            'num_training_events': num_training_events,
            'num_validation_events': num_validation_events,
            'telescope_ids': telescope_ids,
            'num_telescopes': num_telescopes,
            'image_shape': image_shape,
            'num_auxiliary_inputs': 3,
            'num_gamma_hadron_classes': 2
            }
    return metadata
