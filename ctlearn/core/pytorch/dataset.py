""" PyTorch dataset for data loading."""


__all__ = ["PyTorchDataset"]

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dl1_data_handler.reader import ProcessType


class PyTorchDataset(Dataset):
    """
    Generates items/batches for PyTorch application based on DLDataReader.
    """

    def __init__(
        self,
        DLDataReader,
        indices,
        tasks,
        sort_by_intensity=False,
        stack_telescope_images=False,
    ):
        super().__init__()
        self.DLDataReader = DLDataReader
        self.indices = list(indices)
        self.tasks = tasks
        self.sort_by_intensity = sort_by_intensity
        self.stack_telescope_images = stack_telescope_images

        # Convert Keras (H, W, C) input_shape to PyTorch (C, H, W)
        if self.DLDataReader.__class__.__name__ != "DLFeatureVectorReader":
            # 2. Extract base Keras shape safely (H, W, C)
            if self.DLDataReader.mode == "mono":
                keras_shape = self.DLDataReader.input_shape
            elif self.DLDataReader.mode == "stereo":
                first_tel = next(iter(self.DLDataReader.selected_telescopes))
                keras_shape = self.DLDataReader.input_shape[first_tel]                
                # In Keras, 4D input shapes are usually (batch/num_tels, H, W, C)
                if self.stack_telescope_images:
                    num_tels, h, w, c = keras_shape
                    keras_shape = (h, w, num_tels * c)

            # Permute Keras (H, W, C) to PyTorch (C, H, W) via unpacking
            h, w, c = keras_shape
            self.input_shape = (c, h, w)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieves a single data item at the given index.
        Note: If passed a slice/list of indices via custom batching, 
        it falls back to loading as a mini-batch.
        """
        # Support both single index lookup and batch slice lookup
        if isinstance(idx, (int, np.integer)):
            batch_indices = [self.indices[idx]]
        else:
            batch_indices = [self.indices[i] for i in idx]

        if self.DLDataReader.mode == "mono":
            batch = self.DLDataReader.generate_mono_batch(batch_indices)
            features, labels = self._get_mono_item(batch)
        elif self.DLDataReader.mode == "stereo":
            batch = self.DLDataReader.generate_stereo_batch(batch_indices)
            features, labels = self._get_stereo_item(batch)

        # Remove explicit batch dim if caller requested a single index
        if isinstance(idx, (int, np.integer)):
            if isinstance(features, torch.Tensor):
                features = features.squeeze(0)
            if isinstance(labels, dict):
                labels = {k: v.squeeze(0) for k, v in labels.items()}
            elif isinstance(labels, torch.Tensor):
                labels = labels.squeeze(0)

        return features, labels

    def _get_mono_item(self, batch):
        labels = {}
        # Transpose raw batch: (B, H, W, C) -> (B, C, H, W)
        raw_features = torch.from_numpy(batch["features"].data).float()
        features = raw_features.permute(0, 3, 1, 2)

        # Construct task tensors
        if "type" in self.tasks:
            # Return class indices directly as 1D long tensor
            type_labels = torch.from_numpy(batch["true_shower_primary_class"].data).long()
            labels["type"] = type_labels
            if len(self.tasks) == 1:
                labels = type_labels

        if "energy" in self.tasks:
            energy_tensor = torch.from_numpy(batch["log_true_energy"].data).float()
            if isinstance(labels, dict):
                labels["energy"] = energy_tensor
            else:
                labels = energy_tensor

        if "skydirection" in self.tasks:
            sky = np.stack((batch["fov_lon"].data, batch["fov_lat"].data), axis=1)
            sky_tensor = torch.from_numpy(sky).float()
            if isinstance(labels, dict):
                labels["skydirection"] = sky_tensor

        if "cameradirection" in self.tasks:
            cam = np.stack(
                (batch["cam_coord_offset_x"].data, batch["cam_coord_offset_y"].data),
                axis=1,
            )
            cam_tensor = torch.from_numpy(cam).float()
            if isinstance(labels, dict):
                labels["cameradirection"] = cam_tensor

        return features, labels

    def _get_stereo_item(self, batch):
        labels = {}
        if self.DLDataReader.process_type == ProcessType.Simulation:
            batch_grouped = batch.group_by(
                ["obs_id", "event_id", "tel_type_id", "true_shower_primary_class"]
            )
        elif self.DLDataReader.process_type == ProcessType.Observation:
            batch_grouped = batch.group_by(["obs_id", "event_id", "tel_type_id"])

        features, mono_feature_vectors, stereo_feature_vectors = [], [], []
        true_shower_primary_class = []
        log_true_energy = []
        fov_lon, fov_lat = [], []
        cam_coord_offset_x, cam_coord_offset_y = [], []

        for group_element in batch_grouped.groups:
            if "features" in batch.colnames:
                if self.sort_by_intensity:
                    group_element.sort(["hillas_intensity"], reverse=True)
                if self.stack_telescope_images:
                    plain_features = group_element["features"].data
                    stacked_features = np.concatenate(
                        [plain_features[i] for i in range(plain_features.shape[0])],
                        axis=-1,
                    )
                    features.append(stacked_features)
                else:
                    features.append(group_element["features"].data)

            if "mono_feature_vectors" in batch.colnames:
                mono_feature_vectors.append(group_element["mono_feature_vectors"].data)
            if "stereo_feature_vectors" in batch.colnames:
                stereo_feature_vectors.append(
                    group_element["stereo_feature_vectors"].data
                )

            if "type" in self.tasks:
                true_shower_primary_class.append(
                    group_element["true_shower_primary_class"].data[0]
                )
            if "energy" in self.tasks:
                log_true_energy.append(group_element["log_true_energy"].data[0])
            if "skydirection" in self.tasks:
                fov_lon.append(group_element["fov_lon"].data[0])
                fov_lat.append(group_element["fov_lat"].data[0])
            if "cameradirection" in self.tasks:
                cam_coord_offset_x.append(group_element["cam_coord_offset_x"].data)
                cam_coord_offset_y.append(group_element["cam_coord_offset_y"].data)

        # Construct task tensors
        if "type" in self.tasks:
            # 1D Tensor of class indices: shape (batch_size,)
            type_labels = torch.tensor(true_shower_primary_class, dtype=torch.long)
            labels["type"] = type_labels
            if len(self.tasks) == 1:
                labels = type_labels

        if "energy" in self.tasks:
            energy_tensor = torch.tensor(log_true_energy, dtype=torch.float32)
            if isinstance(labels, dict):
                labels["energy"] = energy_tensor

        if "skydirection" in self.tasks:
            sky = np.stack((np.array(fov_lon), np.array(fov_lat)), axis=1)
            sky_tensor = torch.from_numpy(sky).float()
            if isinstance(labels, dict):
                labels["skydirection"] = sky_tensor

        if "cameradirection" in self.tasks:
            cam = np.stack(
                (np.array(cam_coord_offset_x), np.array(cam_coord_offset_y)),
                axis=1,
            )
            cam_tensor = torch.from_numpy(cam).float()
            if isinstance(labels, dict):
                labels["cameradirection"] = cam_tensor

        # Permute feature axes to PyTorch channel conventions
        if "features" in batch.colnames:
            raw_features = torch.tensor(np.array(features), dtype=torch.float32)
            # Shape: (B, H, W, C) -> Permute to (B, C, H, W)
            features = raw_features.permute(0, 3, 1, 2)
        if "mono_feature_vectors" in batch.colnames:
            features = torch.tensor(np.array(mono_feature_vectors), dtype=torch.float32)
        if "stereo_feature_vectors" in batch.colnames:
            features = torch.tensor(np.array(stereo_feature_vectors), dtype=torch.float32)

        return features, labels