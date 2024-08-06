import numpy as np
import tensorflow as tf
import astropy.units as u


class KerasBatchGenerator(tf.keras.utils.Sequence):
    "Generates batches for Keras application"

    def __init__(
        self,
        DLDataReader,
        indices,
        tasks,
        batch_size=64,
        shuffle=True,
        random_seed=1234,
        stack_telescope_images=False,
    ):
        "Initialization"
        self.DLDataReader = DLDataReader
        self.indices = indices
        self.tasks = tasks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.on_epoch_end()
        self.stack_telescope_images = stack_telescope_images

        self.image, self.waveform = False, False
        if self.DLDataReader.image_channels is not None:
            self.image = True
            self.image_shape = self.DLDataReader.image_mapper.image_shapes["LSTCam"]
        if self.DLDataReader.waveform_type is not None:
            self.waveform = True
            self.waveform_shape = self.DLDataReader.waveform_settings["shapes"]["LSTCam"]

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.indices)


    def __getitem__(self, index):
        "Generate one batch of data"
        features, batch = self.DLDataReader.batch_generation(
            self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        )
        if "type" in self.tasks:
            label = tf.keras.utils.to_categorical(
                batch["true_shower_primary_class"].data,
                num_classes=self.DLDataReader.num_classes,
            )
        if "energy" in self.tasks:
            label = batch["log_true_energy"].data
        if "direction" in self.tasks:
            label = np.stack(
                (
                    batch["spherical_offset_az"].data,
                    batch["spherical_offset_alt"].data,
                    batch["angular_separation"].data,
                ),
                axis=1,
            )
        # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
        # https://github.com/keras-team/keras/issues/11735
        return features, label
