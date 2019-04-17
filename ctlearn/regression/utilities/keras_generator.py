import numpy as np
from keras.utils import Sequence


class MAGIC_Generator(Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, include_time=True,
                 position=False, separation=False, energy=False,
                 shuffle=True, apply_log_to_raw=False, cast=False, print_id=False,
                 folder='/data2T/mariotti_data_2/npy_dump/all_npy'):
        'Initialization'
        self.include_time = include_time
        if self.include_time:
            self.dim = (67, 68, 4)
        else:
            self.dim = (67, 68, 2)
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.cast = cast
        self.separation = separation
        self.energy = energy
        self.shuffle = shuffle
        self.folder = folder
        self.position = position
        self.on_epoch_end()
        self.apply_log_to_raw = apply_log_to_raw
        self.select_phe = np.array([False, True, False, True])
        self.print_ID = print_id

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        if self.position:
            y = np.empty((self.batch_size, 2), dtype=float)
        elif self.energy:
            y = np.empty((self.batch_size), dtype=float)
        elif self.separation:
            y = np.empty((self.batch_size), dtype=int)
        else:
            raise ValueError

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.include_time:
                # print(self.folder + '/' + ID + '.npy')
                X[i,] = np.load(self.folder + '/' + ID + '.npy')
            else:
                if self.print_ID:
                    print(ID)
                X[i,] = np.load(self.folder + '/' + ID + '.npy')[:, :, self.select_phe]

            # Store class
            y[i] = self.labels[ID]

            if self.apply_log_to_raw:
                X = np.nan_to_num(np.log10(X))  # do it for avoidin -inf where is 0.

            if self.cast:
                X = X.astype('float16')

        return X, y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
