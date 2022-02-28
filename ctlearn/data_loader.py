import numpy as np
import tensorflow as tf


class KerasBatchGenerator(tf.keras.utils.Sequence):
    'Generates batches for Keras application'
    def __init__(self, DL1DataReaderDL1DH, indices, batch_size=64, mode='train', shuffle=True, concat_telescopes=False):
        'Initialization'
        self.DL1DataReaderDL1DH = DL1DataReaderDL1DH
        self.batch_size = batch_size
        self.indices = indices
        self.mode = mode
        self.shuffle = shuffle
        self.on_epoch_end()
        
        # Decrypt the example description
        # Features
        self.trg_pos, self.trg_shape  = None, None
        self.img_pos, self.img_shape  = None, None
        self.prm_pos, self.prm_shape  = None, None
        # Labels
        self.prt_pos = None
        self.enr_pos = None
        self.drc_pos = None

        for i, desc in enumerate(self.DL1DataReaderDL1DH.example_description):
            if 'trigger' in desc['name']:
                self.trg_pos = i
                self.trg_shape = desc['shape']
            elif 'image' in desc['name']:
                self.img_pos = i
                self.img_shape = desc['shape']
            elif 'parameters' in desc['name']:
                self.prm_pos = i
                self.prm_shape = desc['shape']
            elif 'particletype' in desc['name']:
                self.prt_pos = i
            elif 'energy' in desc['name']:
                self.enr_pos = i
            elif 'direction' in desc['name']:
                self.drc_pos = i

        #Reshape inputs into proper dimensions
        if concat_telescopes:
            self.img_shape = (self.img_shape[1], self.img_shape[2], self.img_shape[0]*self.img_shape[3])


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        return self.__data_generation(self.indices[index*self.batch_size:(index+1)*self.batch_size])

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        'Generates data containing batch_size samples'
        # Initialization
        if self.img_pos is not None:
            images = np.empty((self.batch_size, *self.img_shape))
        if self.prm_pos is not None:
            parameters = np.empty((self.batch_size, *self.prm_shape))

        if self.mode == 'train':
            if self.prt_pos is not None:
                particletype = np.empty((self.batch_size))
            if self.enr_pos is not None:
                energy = np.empty((self.batch_size))
            if self.drc_pos is not None:
                direction = np.empty((self.batch_size, 2))

        # Generate data
        for i, index in enumerate(batch_indices):
            event = self.DL1DataReaderDL1DH[index]
            # Fill the features
            if self.img_pos is not None:
                images[i] = np.reshape(event[self.img_pos], self.img_shape)
            if self.prm_pos is not None:
                parameters[i] = event[self.prm_pos]

            if self.mode == 'train':
                # Fill the labels
                if self.prt_pos is not None:
                    particletype[i] = event[self.prt_pos]
                if self.enr_pos is not None:
                    energy[i] = event[self.enr_pos]
                if self.drc_pos is not None:
                    direction[i] = event[self.drc_pos]

        features = {}
        if self.img_pos is not None:
            features['images'] = images
        if self.prm_pos is not None:
            features['parameters'] = parameters

        labels = {}
        if self.mode == 'train':
            if self.prt_pos is not None:
                labels['particletype'] = tf.keras.utils.to_categorical(particletype, num_classes=2)
                label = tf.keras.utils.to_categorical(particletype, num_classes=2)
            if self.enr_pos is not None:
                labels['energy'] = energy
                label = energy
            if self.drc_pos is not None:
                labels['direction'] = direction
                label = direction

        # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
        # https://github.com/keras-team/keras/issues/11735
        if len(labels) == 1:
            labels = label

        return features, labels
