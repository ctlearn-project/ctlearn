import torch
import numpy as np
from torch.utils.data import Dataset
from .base_loader import BaseDLDataLoader
from dl1_data_handler.reader import ProcessType
from ctlearn.core.ctlearn_enum import Task
from astropy import units as u
import random
import cv2

import concurrent.futures


class PyTorchDLDataLoader(Dataset, BaseDLDataLoader):
 
    def __init__(
        self,
        tasks,
        parameters,
        use_augmentation,
        T=1,
        is_training=False,
        **kwargs,
    ):

        self.is_training=is_training
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._next_batch_future = None

        self.parameter = parameters
        self.use_augmentation = use_augmentation
        self.use_clean = parameters["normalization"]["use_clean"]
        self.use_clean_dvr = parameters["normalization"]["use_clean_dvr"]

        self.task = tasks

        # Augmentation probabilities
        self.mask_augmentation = parameters["augmentation"]["aug_prob"]
        self.aug_prob = parameters["augmentation"]["aug_prob"]
        self.rot_prob = parameters["augmentation"]["rot_prob"]
        self.trans_prob = parameters["augmentation"]["trans_prob"]
        self.flip_hor_prob = parameters["augmentation"]["flip_hor_prob"]
        self.flip_ver_prob = parameters["augmentation"]["flip_ver_prob"]
        self.mask_prob = parameters["augmentation"]["mask_prob"]
        self.mask_dvr_prob = parameters["augmentation"]["mask_dvr_prob"]
        self.noise_prob = parameters["augmentation"]["noise_prob"]
        self.max_aug_rot = parameters["augmentation"]["max_rot"]
        self.max_aug_trans = parameters["augmentation"]["max_trans"]

        # Normalization
        self.type_mu = parameters["normalization"]["type_mu"]
        self.type_sigma = parameters["normalization"]["type_sigma"]
        self.dir_mu = parameters["normalization"]["dir_mu"]
        self.dir_sigma = parameters["normalization"]["dir_sigma"]
        self.energy_mu = parameters["normalization"]["energy_mu"]
        self.energy_sigma = parameters["normalization"]["energy_sigma"]

        super().__init__(**kwargs, tasks=tasks)
        self.on_epoch_end()

        # self.T=T 
        # self.total_len = len(self.indices) * T
        self.set_T(T)
        
        self.hillas_names = [
            "obs_id",
            "event_id",
            "tel_id",
            "hillas_intensity",
            "hillas_skewness",
            "hillas_kurtosis",
            "hillas_fov_lon",
            "hillas_fov_lat",
            "hillas_r",
            "hillas_phi",
            "hillas_length",
            "hillas_length_uncertainty",
            "hillas_width",
            "hillas_width_uncertainty",
            "hillas_psi",
            "timing_intercept",
            "timing_deviation",
            "timing_slope",
            "leakage_pixels_width_1",
            "leakage_pixels_width_2",
            "leakage_intensity_width_1",
            "leakage_intensity_width_2",
            "concentration_cog",
            "concentration_core",
            "concentration_pixel",
            "morphology_n_pixels",
            "morphology_n_islands",
            "morphology_n_small_islands",
            "morphology_n_medium_islands",
            "morphology_n_large_islands",
            "intensity_max",
            "intensity_min",
            "intensity_mean",
            "intensity_std",
            "intensity_skewness",
            "intensity_kurtosis",
            "peak_time_max",
            "peak_time_min",
            "peak_time_mean",
            "peak_time_std",
            "peak_time_skewness",
            "peak_time_kurtosis",
            "core_psi",
        ]

    def set_T(self,T):
        self.T=T 
 
        self.indices = np.tile(self.indices, self.T)
        
    def __len__(self):
        """
        Returns the number of batches per epoch.

        This method calculates the number of batches required to cover the entire dataset
        based on the batch size.

        Returns:
        --------
        int
            Number of batches per epoch.
        """
        return int(np.floor(len(self.indices) / self.batch_size))
 
    
    def on_epoch_end(self):
        """
        Updates indices after each epoch. If a random seed is provided, the indices are shuffled.

        This method is called at the end of each epoch to ensure that the data is shuffled
        if the shuffle attribute is set to True. This helps in improving the training process
        by providing the model with a different order of data in each epoch.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.indices)

    def apply_augmentation(self, image, peak_time):

        for id_batch in range(image.shape[0]):
            random_aug = random.random()

            if random_aug > self.aug_prob:

                if self.task != Task.cameradirection and self.task != Task.skydirection:

                    random_aug_flip_ver = random.random()
                    if random_aug_flip_ver > self.flip_ver_prob:
                        # Vertical flip
                        image[id_batch] = np.expand_dims(cv2.flip(image[id_batch].astype(np.float32), 0), axis=-1)
                        peak_time[id_batch] = np.expand_dims(cv2.flip(peak_time[id_batch].astype(np.float32), 0), axis=-1)

                    random_aug_flip_hor = random.random()
                    if random_aug_flip_hor > self.flip_hor_prob:
                        # Horizontal
                        image[id_batch] = np.expand_dims(cv2.flip(image[id_batch].astype(np.float32), 1), axis=-1)
                        peak_time[id_batch] = np.expand_dims(cv2.flip(peak_time[id_batch].astype(np.float32), 1), axis=-1)
                    # Rotation
                    random_aug_rot = random.random()
                    if random_aug_rot > self.rot_prob:
                        (h, w) = image[id_batch].shape[:2]

                        angle = random.uniform(-self.max_aug_rot, self.max_aug_rot)
                        scale = 1.0  # No scaling
                        center = (w // 2, h // 2)
                        # Step 5: Get the rotation matrix
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

                        # Step 6: Rotate the image
                        image[id_batch] = np.expand_dims(cv2.warpAffine(
                            image[id_batch].astype(np.float32), rotation_matrix, (w, h)
                        ), axis=-1)
                        peak_time[id_batch] = np.expand_dims(cv2.warpAffine(
                            peak_time[id_batch].astype(np.float32), rotation_matrix, (w, h)
                        ), axis=-1)
                    # Translation
                    random_aug_trans = random.random()
                    if random_aug_trans > self.trans_prob:
                        # Translation
                        (h, w) = image[id_batch].shape[:2]

                        tx = random.uniform(-self.max_aug_trans, self.max_aug_trans)
                        ty = random.uniform(-self.max_aug_trans, self.max_aug_trans)
                        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                        image[id_batch] =  np.expand_dims(cv2.warpAffine(
                            image[id_batch].astype(np.float32), translation_matrix, (w, h)
                        ), axis=-1)
                        peak_time[id_batch] =  np.expand_dims(cv2.warpAffine(
                            peak_time[id_batch].astype(np.float32), translation_matrix, (w, h)
                        ), axis=-1)
                else:
                    doNothing = True

        return image, peak_time

    def _fetch_batch(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        if len(batch_indices) == 0:
            raise IndexError(f"No data for batch index {index} (batch_indices empty)")
        
        if self.DLDataReader.mode == "mono":
            batch = self.DLDataReader.generate_mono_batch(batch_indices)
            features, labels = self._get_mono_item(batch)
        elif self.DLDataReader.mode == "stereo":
            batch = self.DLDataReader.generate_stereo_batch(batch_indices)
            features, labels = self._get_stereo_item(batch)
        return features, labels
    
    # Fetching batches
    def __getitem__(self, index):
   
        data_idx = index
        # data_idx = index % int((self.total_len/self.T)/self.batch_size)
        t = index // int(np.ceil(len(self.indices)/self.T/self.batch_size))
   
        # If this is the first call, fetch synchronously, and schedule the next
        if self._next_batch_future is None:
            features, labels = self._fetch_batch(data_idx)
        else:
            features, labels = self._next_batch_future.result()  # Wait for the prefetch to finish

        # Schedule the next batch prefetch
        if data_idx + 1 < len(self):
            self._next_batch_future = self.executor.submit(self._fetch_batch, data_idx + 1)
        else:
            self._next_batch_future = None  # No more batches

        return features, labels, t

    # def __getitem__(self, index):
    #     """
    #     Generate one batch of data and retrieve the features and labels.

    #     This method is called to generate one batch of monoscopic and stereoscopic data based on
    #     the index provided. It calls either _get_mono_item(batch) or _get_stereo_item(batch)
    #     based on the mode of the DLDataReader.

    #     Parameters:
    #     -----------
    #     index : int
    #         Index of the batch to generate.

    #     Returns:
    #     --------
    #     tuple
    #         A tuple containing the input data as features and the corresponding labels.
    #     """

    #     # data_idx = index
    #     # data_idx = index % int((self.total_len/self.T)/self.batch_size)
    #     t = index // int(np.ceil(len(self.indices)/self.T/self.batch_size))
    #     # Generate indices of the batch
    #     batch_indices = self.indices[
    #         index * self.batch_size : (index + 1) * self.batch_size
    #     ]
    #     features, labels = None, None
    #     if self.DLDataReader.mode == "mono":
    #         batch = self.DLDataReader.generate_mono_batch(batch_indices)
    #         features, labels = self._get_mono_item(batch)
    #     elif self.DLDataReader.mode == "stereo":
    #         batch = self.DLDataReader.generate_stereo_batch(batch_indices)
    #         features, labels = self._get_stereo_item(batch)

    #     return features, labels, t

    # def cam_to_alt_az(
    #     self, tel_id, focal_length, pix_rotation, tel_az, tel_alt, cam_x, cam_y
    # ):
    #     """
    #     Transform camera coordinate offsets (cam_x, cam_y) into Alt/Az sky coordinates.

    #     This method converts the given camera coordinates for each telescope into sky coordinates
    #     (Altitude and Azimuth), using the known pointing of each telescope and camera geometry
    #     such as focal length and pixel rotation.

    #     Parameters
    #     ----------
    #     tel_id : list or array-like
    #         List of telescope IDs corresponding to each event or observation.

    #     focal_length : list or array-like
    #         Focal length of the telescopes in meters.

    #     pix_rotation : list or array-like
    #         Pixel rotation angles (in degrees) for each telescope camera.

    #     tel_az : list or array-like
    #         Azimuth of telescope pointing (in radians).

    #     tel_alt : list or array-like
    #         Altitude of telescope pointing (in radians).

    #     cam_x : list or array-like
    #         Camera x-coordinate positions (in meters).

    #     cam_y : list or array-like
    #         Camera y-coordinate positions (in meters).

    #     Returns
    #     -------
    #     sky_coords_alt : list
    #         List of reconstructed Altitude coordinates (in degrees).

    #     sky_coords_az : list
    #         List of reconstructed Azimuth coordinates (in degrees).
    #     """
    #     from astropy.time import Time

    #     LST_EPOCH = Time("2018-10-01T00:00:00", scale="utc")
    #     from astropy.coordinates import AltAz, SkyCoord
    #     from ctapipe.coordinates import CameraFrame
    #     from astropy import units as u

    #     # # Get telescope ground frame position
    #     tel_ground_frame = self.DLDataReader.subarray.tel_coords[
    #         self.DLDataReader.subarray.tel_ids_to_indices(tel_id)
    #     ]

    #     # AltAz frame setup
    #     altaz = AltAz(
    #         location=tel_ground_frame.to_earth_location(),
    #         obstime=LST_EPOCH,
    #     )

    #     # Telescope pointing SkyCoord
    #     fix_tel_pointing = SkyCoord(
    #         az=tel_az * u.rad,
    #         alt=tel_alt * u.rad,
    #         frame=altaz,
    #     )

    #     sky_coords_alt = []
    #     sky_coords_az = []

    #     for id in range(len(focal_length)):

    #         camera_frame = CameraFrame(
    #             focal_length=focal_length[id] * u.m,
    #             rotation=pix_rotation[id] * u.deg,
    #             telescope_pointing=fix_tel_pointing[id],
    #         )

    #         cam_coord = SkyCoord(
    #             x=cam_x[id] * u.m, y=cam_y[id] * u.m, frame=camera_frame
    #         )

    #         sky_coord = cam_coord.transform_to(altaz[id])

    #         sky_coords_alt.append(sky_coord.alt.to_value(u.deg).item())
    #         sky_coords_az.append(sky_coord.az.to_value(u.deg).item())

    #     return sky_coords_alt, sky_coords_az
    def cam_to_alt_az(self, tel_id, focal_length, pix_rotation, tel_az, tel_alt, cam_x, cam_y):

        from astropy.time import Time
        from astropy.coordinates import AltAz, SkyCoord
        from ctapipe.coordinates import CameraFrame
        from astropy import units as u

        LST_EPOCH = Time("2018-10-01T00:00:00", scale="utc")
        sky_coords_alt = []
        sky_coords_az = []

        for id in range(len(focal_length)):
            # Telescopio correspondiente
            tel_ground_frame = self.DLDataReader.subarray.tel_coords[
                self.DLDataReader.subarray.tel_ids_to_indices(tel_id[id])
            ]

            # Frame AltAz particular para cada telescopio
            altaz_frame = AltAz(
                location=tel_ground_frame.to_earth_location(),
                obstime=LST_EPOCH,
            )

            fix_tel_pointing = SkyCoord(
                az=tel_az[id] * u.rad,
                alt=tel_alt[id] * u.rad,
                frame=altaz_frame,
            )

            camera_frame = CameraFrame(
                focal_length=focal_length[id] * u.m,
                rotation=pix_rotation[id] * u.deg,
                telescope_pointing=fix_tel_pointing,
            )

            cam_coord = SkyCoord(
                x=cam_x[id] * u.m, y=cam_y[id] * u.m, frame=camera_frame
            )

            # Transformar correctamente
            sky_coord = cam_coord.transform_to(altaz_frame)

            sky_coords_alt.append(sky_coord.alt.to_value(u.deg).item())
            sky_coords_az.append(sky_coord.az.to_value(u.deg).item())

        return sky_coords_alt, sky_coords_az

    def _get_mono_item(self, batch):
        """
        Retrieve the features and labels for one batch of monoscopic data.

        This method is called to retrieve the features and labels for one batch of
        monoscopic data. The labels are set up based on the tasks specified.

        Parameters:
        -----------
        batch : astropy.table.Table
            A table containing the data for the batch.

        Returns:
        --------
        tuple
            A tuple containing the input data as features and the corresponding labels.
        """
        # Retrieve the telescope images and store in the features dictionary
        labels = {}
        features = {"input": batch["features"].data}
        
        if "type" in self.tasks:
            labels["type"] = np.stack(batch["true_shower_primary_class"].data)

        labels["energy"] = batch["log_true_energy"].data

        if "skydirection" in self.tasks:
            labels["skydirection"] = np.stack(
                (
                    batch["fov_lon"].data,
                    batch["fov_lat"].data,
                    batch["angular_separation"].data,
                ),
                axis=1,
            )
        if "cameradirection" in self.tasks:
            labels["cameradirection"] = np.stack(
                (
                    batch["cam_coord_offset_x"].data,
                    batch["cam_coord_offset_y"].data,
                    batch["cam_coord_distance"].data,
                ),
                axis=1,
            )

        if "skydirection" in labels.keys():
            labels["direction"] = labels["skydirection"]

        if "cameradirection" in labels.keys():
            labels["direction"] = labels["cameradirection"]

        # features["hillas"] = self.DLDataReader.get_parameters(batch, self.hillas_names)
        features["hillas"] = self.DLDataReader.get_parameters(batch)

        image = features["input"][..., 0:1]
        peak_time = features["input"][..., 1:2]


        # ----------------------------------------------------
        # Remove negative numbers and avoid inf or nans
        # ----------------------------------------------------
        image[image < 0] = 0
        peak_time[peak_time < 0] = 0
        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 0
        peak_time[np.isnan(peak_time)] = 0
        peak_time[np.isinf(peak_time)] = 0


        # image, peak_time = self.apply_augmentation(image, peak_time)

        # image = np.transpose(image, (0, 3, 1, 2))
        # peak_time = np.transpose(peak_time, (0, 3, 1, 2))

        if self.task == Task.type:
            image = (image - self.type_mu) / self.type_sigma
            peak_time = (peak_time - self.type_mu) / self.type_sigma

        if self.task == Task.energy:
            image = (image - self.energy_mu) / self.energy_sigma
            peak_time = (peak_time - self.energy_mu) / self.energy_sigma

        if self.task == Task.cameradirection or self.task == Task.skydirection:
            image = (image - self.dir_mu) / self.dir_sigma
            peak_time = (peak_time - self.dir_mu) / self.dir_sigma

        # image = torch.from_numpy(image).contiguous().float()
        # peak_time = torch.from_numpy(peak_time).contiguous().float()

        features_out = {}
        features_out["image"] = image
        features_out["peak_time"] = peak_time
        
        features_out["image"] = torch.from_numpy(image).float().permute(0, 3, 1, 2).contiguous()
        features_out["peak_time"] = torch.from_numpy(peak_time).float().permute(0, 3, 1, 2).contiguous()
        
        for key in labels.keys():
            labels[key] = torch.from_numpy(labels[key]).contiguous()

            if key != "type":
                labels[key] = labels[key].unsqueeze(-1)

        for key in features["hillas"].keys():
            features["hillas"][key] = (
                torch.from_numpy(np.array(features["hillas"][key]))
                .contiguous()
                .unsqueeze(-1)
            )

        if "cameradirection" in self.tasks:

            tel_ids = batch["tel_id"].data

            # tel_ground_frame = self.DLDataReader.subarray.tel_coords[
            #     self.DLDataReader.subarray.tel_ids_to_indices(tel_ids)
            # ]

            focal_lengths = [
                self.DLDataReader.subarray.tel[
                    tel_id
                ].camera.geometry.frame.focal_length
                for tel_id in tel_ids
            ]
            pix_rotations = [
                self.DLDataReader.pix_rotation[tel_id] for tel_id in tel_ids
            ]

            labels["focal_length"] = np.array(
                [focal.to_value(u.m) for focal in focal_lengths]
            )
            labels["pix_rotation"] = np.array(
                [rot.to_value(u.deg) for rot in pix_rotations]
            )
            # labels["tel_ground"] = tel_ground_frame
            labels["tel_ids"] = tel_ids
            labels["true_alt"] = np.array([val for val in batch["true_alt"]])
            labels["true_az"] = np.array([val for val in batch["true_az"]])
            labels["tel_az"] = batch["telescope_pointing_azimuth"].data
            labels["tel_alt"] = batch["telescope_pointing_altitude"].data

            # cam_x = labels["cameradirection"][:,0].cpu().numpy().squeeze(-1)
            # cam_y = labels["cameradirection"][:,1].cpu().numpy().squeeze(-1)

            # sky_coords_alt, sky_coords_az = self.cam_to_alt_az(labels["tel_ids"], labels["focal_length"], labels["pix_rotation"],labels["tel_az"],labels["tel_alt"], cam_x, cam_y)


                
        if self.is_training:
            N = 4  # Repeating the number of high energies 
            features_out["hillas"] = features["hillas"]
            #-------------------------------------------------
            if self.use_augmentation:
                energy_log = torch.pow(10,labels["energy"].squeeze(-1))  # shape [N]
                high_energy_mask = energy_log > 1  # log10(E/TeV) > 0 => E > 1 TeV

                idx_to_duplicate = torch.where(high_energy_mask)[0]

                if len(idx_to_duplicate) > 0:
                    def duplicate_tensor(t,idx_to_duplicate):
                        if isinstance(t, torch.Tensor):
                            extra = torch.cat([t[idx_to_duplicate] for _ in range(N)], dim=0)
                            return torch.cat([t, extra], dim=0).contiguous()
                        elif isinstance(t, np.ndarray):
                            # Si t es 1D, t[idx_to_duplicate] ya es de shape (M,), sólo hace falta stackear
                            # extra = np.tile(t[idx_to_duplicate], N)
                            # return np.concatenate([t, extra], axis=0)
                            idx_to_duplicate = idx_to_duplicate.cpu().numpy() if hasattr(idx_to_duplicate, "cpu") else idx_to_duplicate
                            # extra = np.tile(t[idx_to_duplicate], (N, 1, 1, 1))  # si shape es (n, x, y, z)
                            # Mejor: stack y luego reshape
                            extra = np.concatenate([t[idx_to_duplicate] for _ in range(N)], axis=0)
                            # O si es 1D, puedes hacer
                            # extra = np.tile(t[idx_to_duplicate], N)
                            return np.concatenate([t, extra], axis=0)
                                
                        else:
                            raise TypeError(f"Data type not supported: {type(t)}")
                        
                    # Duplica todas las features principales
                    for key in features_out:
                        if isinstance(features_out[key], dict):
                            # Por ejemplo, hillas es un dict de tensores
                            for k in features_out[key]:
                                features_out[key][k] = duplicate_tensor(features_out[key][k],idx_to_duplicate)
                        else:
                            features_out[key] = duplicate_tensor(features_out[key],idx_to_duplicate)

                    # Duplica las labels
                    for key in labels:
                        labels[key] = duplicate_tensor(labels[key],idx_to_duplicate)


        if self.use_augmentation:
            
            if isinstance(features_out["image"], torch.Tensor):
                features_out["image"] = features_out["image"].cpu().numpy()
            if isinstance(features_out["peak_time"], torch.Tensor):
                features_out["peak_time"] = features_out["peak_time"].cpu().numpy()
            
            image, peak_time = self.apply_augmentation(features_out["image"], features_out["peak_time"])

            
        image = np.transpose(image, (0, 3, 1, 2))
        peak_time = np.transpose(peak_time, (0, 3, 1, 2))

        features_out["image"] = torch.from_numpy(image.copy()).contiguous().float()
        features_out["peak_time"] = torch.from_numpy(peak_time.copy()).contiguous().float()
        

            

        if not self.is_training:
            # Generate keep_idx as before
            hillas = features["hillas"]
            leakage = np.array(hillas["leakage_intensity_width_2"])
            intensity = np.array(hillas["hillas_intensity"])
            keep_idx = np.where((leakage < 0.2) & (intensity > 50))[0]
            # keep_idx = np.where((leakage > 0.8) & (intensity > 50))[0]

        # Filter features_out
        for key in features_out:
            features_out[key] = features_out[key][keep_idx]
        
        features_out["hillas"] = features["hillas"]

        for key in features["hillas"]:
            features_out["hillas"][key] = (features["hillas"][key])[keep_idx]
            
        # Filter labels (since it's a dict too)
        for key in labels:
            labels[key] = labels[key][keep_idx]
    
        return features_out, labels

    # TODO: Not adapted to pytorch
    def _get_stereo_item(self, batch):
        """
        Retrieve the features and labels for one batch of stereoscopic data.

        This method is called to retrieve the features and labels for one batch of
        stereoscopic data. The original batch is grouped to retrieve the telescope
        data for each event and then the telescope images or waveforms are stored
        by the hillas intensity or stacked if required. Feature vectors can also
        be retrieved if available for ``telescope``- and ``subarray``level. The
        labels are set up based on the tasks specified.

        Parameters:
        -----------
        batch : astropy.table.Table
            A table containing the data for the batch.

        Returns:
        --------
        tuple
            A tuple containing the input data as features and the corresponding labels.
        """
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
        fov_lon, fov_lat, angular_separation = [], [], []
        cam_coord_offset_x, cam_coord_offset_y, cam_coord_distance = [], [], []
        for group_element in batch_grouped.groups:
            if "features" in batch.colnames:
                if self.sort_by_intensity:
                    # Sort images by the hillas intensity in a given batch if requested
                    group_element.sort(["hillas_intensity"], reverse=True)
                # Stack the telescope images for stereo analysis
                if self.stack_telescope_images:
                    # Retrieve the telescope images
                    plain_features = group_element["features"].data
                    # Stack the telescope images along the last axis
                    stacked_features = np.concatenate(
                        [plain_features[i] for i in range(plain_features.shape[0])],
                        axis=-1,
                    )
                    # Append the stacked images to the features list
                    # shape: (batch_size, image_shape, image_shape, n_channels * n_tel)
                    features.append(stacked_features)
                else:
                    # Append the plain images to the features list
                    # shape: (batch_size, n_tel, image_shape, image_shape, n_channels)
                    features.append(group_element["features"].data)
            # Retrieve the feature vectors
            if "mono_feature_vectors" in batch.colnames:
                mono_feature_vectors.append(group_element["mono_feature_vectors"].data)
            if "stereo_feature_vectors" in batch.colnames:
                stereo_feature_vectors.append(
                    group_element["stereo_feature_vectors"].data
                )
            # Retrieve the labels for the tasks
            # FIXME: This won't work for divergent pointing directions
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
        # Store the labels in the labels dictionary
        if "type" in self.tasks:
            labels["type"] = np.array(true_shower_primary_class)

            # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
            # https://github.com/keras-team/keras/issues/11735
            if len(self.tasks) == 1:
                labels = np.array(true_shower_primary_class)

        if "energy" in self.tasks:
            labels["energy"] = np.array(log_true_energy)
        if "skydirection" in self.tasks:
            labels["skydirection"] = np.stack(
                (
                    np.array(fov_lon),
                    np.array(fov_lat),
                ),
                axis=1,
            )
        if "cameradirection" in self.tasks:
            labels["cameradirection"] = np.stack(
                (
                    np.array(cam_coord_offset_x),
                    np.array(cam_coord_offset_y),
                ),
                axis=1,
            )
        # Store the fatures in the features dictionary
        if "features" in batch.colnames:
            features = {"input": np.array(features)}
        # TDOO: Add support for both feature vectors
        if "mono_feature_vectors" in batch.colnames:
            features = {"input": np.array(mono_feature_vectors)}
        if "stereo_feature_vectors" in batch.colnames:
            features = {"input": np.array(stereo_feature_vectors)}

        image = features[:, :, :, 0]
        peak_time = features[:, :, :, 1]

        image = np.transpose(image, (2, 0, 1))
        peak_time = np.transpose(peak_time, (2, 0, 1))

        features_out = None
        features_out["image"] = image
        features_out["peak_time"] = peak_time
        return features_out, labels

    # Include _get_mono_item and _get_stereo_item as needed
