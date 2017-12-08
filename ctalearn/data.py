from operator import itemgetter

import tables
import numpy as np

def load_HDF5_data(filename, index, auxiliary_data, metadata,
        sort_telescopes_by_trigger=False, mode='TRAIN'):

    # Read the data at the given table and index from the file
    with tables.open_file(filename, mode='r') as f:
        if mode == 'TRAIN':
            table = f.root.E0.Events_Training
        elif mode == 'VALID':
            table = f.root.E0.Events_Validation
        else:
            raise ValueError("Mode must be 'TRAIN' or 'VALID'")
        record = table.read(index, index + 1)
        
        # Get classification label by converting CORSIKA particle code
        gamma_hadron_label = record['gamma_hadron_label'][0]
        if gamma_hadron_label == 0: # gamma ray
            gamma_hadron_label = 1
        elif gamma_hadron_label == 101: # proton
            gamma_hadron_label = 0
        
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
    
    # Get binary values indicating whether each telescope triggered
    telescope_triggers = [0 if i < 0 else 1 for i in image_indices]

    telescope_positions = auxiliary_data['telescope_positions']

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

def load_HDF5_auxiliary_data(filename):
  
    telescope_positions = []
    with tables.open_file(filename, mode='r') as f:
        for row in f.root.Tel_Table.iterrows():
            telescope_positions.append(row["tel_x"])
            telescope_positions.append(row["tel_y"])
            telescope_positions.append(row["tel_z"])
    auxiliary_data = {
            'telescope_positions': telescope_positions
            }
    
    return auxiliary_data

def load_HDF5_metadata(filename):
   
    with tables.open_file(filename, mode='r') as f:
        num_training_events = f.root.E0.Events_Training.shape[0]
        num_validation_events = f.root.E0.Events_Validation.shape[0]
        # List of telescope IDs ordered by mapping index
        telescope_ids = ["T" + str(row["tel_id"]) for row 
                in f.root.Tel_Table.iterrows()]
        num_telescopes = f.root.Tel_Table.shape[0]
        # All telescope images have the same shape
        image_shape = f.root.E0._f_get_child(telescope_ids[0]).shape[1:]
    metadata = {
            'num_training_events': num_training_events,
            'num_validation_events': num_validation_events,
            'telescope_ids': telescope_ids,
            'num_telescopes': num_telescopes,
            'image_shape': image_shape,
            'num_auxiliary_inputs': 3,
            'num_gamma_hadron_classes': 2,
            }
    return metadata
