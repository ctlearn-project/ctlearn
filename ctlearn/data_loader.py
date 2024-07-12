import numpy as np
import tensorflow as tf


class KerasBatchGenerator(tf.keras.utils.Sequence):
    "Generates batches for Keras application"

    def __init__(
        self,
        DLDataReader,
        indices,
        batch_size=64,
        mode="train",
        class_names=None,
        shuffle=True,
        stack_telescope_images=False,
    ):
        "Initialization"
        self.DLDataReader = DLDataReader
        self.batch_size = batch_size
        self.indices = indices
        self.mode = mode
        self.class_names = class_names
        self.shuffle = shuffle
        self.stack_telescope_images = stack_telescope_images
        self.on_epoch_end()

        # Decrypt the example description
        # Features
        self.singleimg_shape = None
        self.trg_pos, self.trg_shape = None, None
        self.trgpatch_pos, self.trgpatch_shape = None, None
        self.wvf_pos, self.wvf_shape = None, None
        self.img_pos, self.img_shape = None, None
        self.prm_pos, self.prm_shape = None, None
        self.parameter_list = []
        # Additional info
        self.evt_pos, self.obs_pos = None, None
        self.event_list, self.obs_list = [], []
        # Labels
        self.prt_pos, self.enr_pos, self.drc_pos = None, None, None
        self.drc_unit = None
        self.prt_labels = []
        self.enr_labels = []
        self.az_labels, self.alt_labels, self.sep_labels = [], [], []
        self.trgpatch_labels = []
        self.energy_unit = None

        for i, desc in enumerate(self.DLDataReader.example_description):
            if "HWtrigger" in desc["name"]:
                self.trg_pos = i
                self.trg_shape = desc["shape"]
            elif "waveform" in desc["name"]:
                self.wvf_pos = i
                self.wvf_shape = desc["shape"]
            elif "trigger_patch" in desc["name"]:
                self.trgpatch_pos = i
                self.trgpatch_shape = desc["shape"]
            elif "image" in desc["name"]:
                self.img_pos = i
                self.img_shape = desc["shape"]
            elif "parameters" in desc["name"]:
                self.prm_pos = i
                self.prm_shape = desc["shape"]
            elif "true_shower_primary_id" in desc["name"]:
                self.prt_pos = i
            elif "energy" in desc["name"]:
                self.enr_pos = i
                self.energy_unit = desc["unit"]
            elif "direction" in desc["name"]:
                self.drc_pos = i
                self.drc_unit = desc["unit"]
            elif "event_id" in desc["name"]:
                self.evt_pos = i
            elif "obs_id" in desc["name"]:
                self.obs_pos = i

        # Retrieve shape from a single image in stereo analysis
        if self.trg_pos is not None and self.img_pos is not None:
            self.singleimg_shape = (
                self.img_shape[1],
                self.img_shape[2],
                self.img_shape[3],
            )
        # Reshape inputs into proper dimensions for the stereo analysis with merged models
        if self.stack_telescope_images:
            self.img_shape = (
                self.img_shape[1],
                self.img_shape[2],
                self.img_shape[0] * self.img_shape[3],
            )

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        return self.__data_generation(
            self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        )

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        "Generates data containing batch_size samples"
        if self.trg_pos is not None:
            triggers = np.empty((self.batch_size, *self.trg_shape))
        if self.wvf_pos is not None:
            waveforms = np.empty((self.batch_size, *self.wvf_shape))
        if self.img_pos is not None:
            images = np.empty((self.batch_size, *self.img_shape))
        if self.prm_pos is not None:
            parameters = np.empty((self.batch_size, *self.prm_shape))

        if self.mode == "train":
            if self.prt_pos is not None:
                particletype = np.empty((self.batch_size))
            if self.enr_pos is not None:
                energy = np.empty((self.batch_size))
            if self.drc_pos is not None:
                direction = np.empty((self.batch_size, 3))
            if self.trgpatch_pos is not None:
                trigger_patches_true_image_sum = np.empty(
                    (self.batch_size, *self.trgpatch_shape)
                )
        # Generate data
        for i, index in enumerate(batch_indices):
            event = self.DLDataReader[index]
            # Fill the features
            if self.trg_pos is not None:
                triggers[i] = event[self.trg_pos]
            if self.wvf_pos is not None:
                waveforms[i] = event[self.wvf_pos]
            if self.img_pos is not None:
                images[i] = np.reshape(event[self.img_pos], self.img_shape)
            if self.prm_pos is not None:
                parameters[i] = event[self.prm_pos]

            if self.mode == "train":
                # Fill the labels
                if self.prt_pos is not None:
                    particletype[
                        i
                    ] = self.DLDataReader.shower_primary_id_to_class[
                        int(event[self.prt_pos])
                    ]
                if self.enr_pos is not None:
                    energy[i] = event[self.enr_pos]
                if self.drc_pos is not None:
                    direction[i] = event[self.drc_pos]
                if self.trgpatch_pos is not None:
                    trigger_patches_true_image_sum[i] = event[self.trgpatch_pos]
            else:
                # Save all labels for the prediction phase
                if self.prt_pos is not None:
                    self.prt_labels.append(np.float32(event[self.prt_pos]))
                if self.enr_pos is not None:
                    self.enr_labels.append(np.float32(event[self.enr_pos]))
                if self.drc_pos is not None:
                    self.az_labels.append(np.float32(event[self.drc_pos][0]))
                    self.alt_labels.append(np.float32(event[self.drc_pos][1]))
                    self.sep_labels.append(np.float32(event[self.drc_pos][2]))
                if self.trgpatch_pos is not None:
                    self.trgpatch_labels.append(np.float32(event[self.trgpatch_pos]))
                # Save all parameters for the prediction phase
                if self.prm_pos is not None:
                    self.parameter_list.append(event[self.prm_pos])
                # Save event and obs id
                if self.evt_pos is not None:
                    self.event_list.append(np.float32(event[self.evt_pos]))
                if self.obs_pos is not None:
                    self.obs_list.append(np.float32(event[self.obs_pos]))

        features = {}
        if self.trg_pos is not None:
            features["triggers"] = triggers
        if self.wvf_pos is not None:
            features["waveforms"] = waveforms
        if self.img_pos is not None:
            features["images"] = images
        if self.prm_pos is not None:
            features["parameters"] = parameters

        labels = {}
        if self.mode == "train":
            if self.prt_pos is not None:
                labels["type"] = tf.keras.utils.to_categorical(
                    particletype,
                    num_classes=self.DLDataReader.num_classes,
                )
                label = tf.keras.utils.to_categorical(
                    particletype,
                    num_classes=self.DLDataReader.num_classes,
                )
            if self.enr_pos is not None:
                labels["energy"] = energy.reshape((-1, 1))
                label = energy.reshape((-1, 1))
            if self.drc_pos is not None:
                labels["direction"] = direction
                label = direction
            if self.trgpatch_pos is not None and self.DLDataReader.reco_cherenkov_photons:
                labels["cherenkov_photons"] = trigger_patches_true_image_sum
                label = trigger_patches_true_image_sum

        # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
        # https://github.com/keras-team/keras/issues/11735
        if len(labels) == 1:
            labels = label

        return features, labels
