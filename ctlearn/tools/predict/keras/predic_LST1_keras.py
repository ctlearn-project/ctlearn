from ctapipe.io import read_table
from astropy.table import join
import keras
from dl1_data_handler.reader import (
    get_unmapped_image
)
import numpy as np

def predictions(self):
    event_id, tel_azimuth, tel_altitude, trigger_time = [], [], [], []
    prediction, energy, cam_coord_offset_x, cam_coord_offset_y = [], [], [], []
    classification_fvs, energy_fvs, direction_fvs = [], [], []
    for start in range(0, self.table_length, self.batch_size):
        stop = min(start + self.batch_size, self.table_length)
        self.log.debug("Processing chunk from '%d' to '%d'.", start, stop - 1)
        # Read the data
        dl1_table = read_table(
            self.input_url, self.image_table_path, start=start, stop=stop
        )
        # Join the dl1 table with the parameter table to perform quality selection
        dl1_table = join(
            left=dl1_table,
            right=self.parameter_table,
            keys=["event_id"],
        )
        dl1_table = join(
            left=dl1_table,
            right=self.trigger_table,
            keys=["event_id"],
        )
        # Initialize a boolean mask to True for all events in the sliced dl1 table
        passes_quality_checks = np.ones(len(dl1_table), dtype=bool)
        # Quality selection based on the dl1b parameter
        if self.quality_query:
            passes_quality_checks = self.quality_query.get_table_mask(dl1_table)
        # Apply the mask to filter events that are not fufilling the quality criteria
        dl1_table = dl1_table[passes_quality_checks]
        if len(dl1_table) == 0:
            self.log.debug("No events passed the quality selection.")
            continue
        data = []
        for event in dl1_table:
            # Get the unmapped image
            image = get_unmapped_image(event, self.channels, self.transforms)
            data.append(self.image_mapper.map_image(image))
        input_data = {"input": np.array(data)}
        # Temp fix for supporting keras2 & keras3
        if int(keras.__version__.split(".")[0]) >= 3:
            input_data = input_data["input"]

        event_id.extend(dl1_table["event_id"].data)
        tel_azimuth.extend(dl1_table["tel_az"].data)
        tel_altitude.extend(dl1_table["tel_alt"].data)
        trigger_time.extend(dl1_table["time"].mjd)
        if self.load_type_model_from is not None:
            classification_feature_vectors = self.backbone_type.predict_on_batch(input_data)
            classification_fvs.extend(classification_feature_vectors)
            predict_data = self.head_type.predict_on_batch(classification_feature_vectors)
            prediction.extend(predict_data[:, 1])
        if self.load_energy_model_from is not None:
            energy_feature_vectors = self.backbone_energy.predict_on_batch(input_data)
            energy_fvs.extend(energy_feature_vectors)
            predict_data = self.head_energy.predict_on_batch(energy_feature_vectors)
            energy.extend(predict_data.T[0])
        if self.load_cameradirection_model_from is not None:
            direction_feature_vectors = self.backbone_direction.predict_on_batch(input_data)
            direction_fvs.extend(direction_feature_vectors)
            predict_data = self.head_direction.predict_on_batch(direction_feature_vectors)
            cam_coord_offset_x.extend(predict_data.T[0])
            cam_coord_offset_y.extend(predict_data.T[1])
            
    return event_id, tel_azimuth, tel_altitude, trigger_time, prediction, energy, cam_coord_offset_x, cam_coord_offset_y, classification_fvs, energy_fvs, direction_fvs