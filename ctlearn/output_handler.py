import logging
import numpy as np
import pandas as pd

def write_output(h5file, reader, indices, example_description, predictions, prediction_label='prediction'):
    data = {}
    tel_pointing = np.array([0.0, 0.0], np.float32)
    if reader.pointing_mode == 'fix_subarray':
        tel_pointing = reader.pointing
    energy_unit = 'TeV'
    for i, idx in enumerate(indices):
        for val, des in zip(reader[idx], example_description):
            if des['name'] == 'pointing':
                tel_pointing = val
            elif des['name'] == 'trigger_time':
                if i == 0:
                    data['time'] = []
                data['time'].append(val)
            elif des['name'] == 'particletype':
                if i == 0:
                    data['mc_particle'] = []
                data['mc_particle'].append(val)
            elif des['name'] == 'energy':
                if i == 0:
                    data['mc_energy'] = []
                if des['unit'] == 'log(TeV)':
                    energy_unit = 'log(TeV)'
                    val[0] = np.power(10,val[0])
                data['mc_energy'].append(val[0])
            elif des['name'] in ['direction', 'delta_direction'] :
                if i == 0:
                    data['mc_altitude'], data['mc_azimuth'] = [],[]
                data['mc_altitude'].append(val[0] + tel_pointing[0])
                data['mc_azimuth'].append(val[1] + tel_pointing[1])
            elif des['name'] == 'impact':
                if i == 0:
                    data['mc_impact_x'], data['mc_impact_y'] = [],[]
                data['mc_impact_x'].append(val[0])
                data['mc_impact_y'].append(val[1])
            elif des['name'] == 'event_id':
                if i == 0:
                    data['event_id'] = []
                data['event_id'].append(val)
            elif des['name'] == 'core_y':
                if i == 0:
                    data['core_y'] = []
                data['core_y'].append(val)
            elif des['name'] == 'core_x':
                if i == 0:
                    data['core_x'] = []
                data['core_x'].append(val)
            elif des['name'] == 'mc_energy':
                if i == 0:
                    data['mc_energy'] = []
                data['mc_energy'].append(val)

            if des['name'].endswith('parameters'):
                if i == 0:
                    data['parameters'] = []
                data['parameters'].append(val)
            #else:
            #    if des['name'] in reader.event_info:
            #        print(des['name'])
            #        if i == 0:
            #            data[des['name']] = []
            #        data[des['name']].append(val)
            #        print(data[des['name']])

        prediction = predictions[i]
        # Gamma/hadron classification
        if 'particletype' in prediction and 'particletype_probabilities' in prediction:
            if i == 0:
                data['reco_particle'], data['reco_gammaness'] = [],[]
            data['reco_particle'].append(prediction['particletype'])
            data['reco_gammaness'].append(prediction['particletype_probabilities'][1])
        # Energy regression
        if 'energy' in prediction:
            if i == 0:
                data['reco_energy'] = []
            if energy_unit == 'log(TeV)':
                prediction['energy'][0] = np.power(10,prediction['energy'][0])
            data['reco_energy'].append(prediction['energy'][0])
        # Arrival direction regression
        if 'direction' in prediction:
            if i == 0:
                data['reco_altitude'], data['reco_azimuth'] = [],[]
            data['reco_altitude'].append(prediction['direction'][0] + tel_pointing[0])
            data['reco_azimuth'].append(prediction['direction'][1] + tel_pointing[1])
        if 'delta_direction' in prediction:
            if i == 0:
                data['reco_altitude'], data['reco_azimuth'] = [],[]
            data['reco_altitude'].append(prediction['delta_direction'][0] + tel_pointing[0])
            data['reco_azimuth'].append(prediction['delta_direction'][1] + tel_pointing[1])
        # Impact parameter regression
        if 'impact' in prediction:
            if i == 0:
                data['reco_impact_x'], data['reco_impact_y'] = [],[]
            data['reco_impact_x'].append(prediction['impact'][0])
            data['reco_impact_y'].append(prediction['impact'][1])
        # Shower maximum regression
        if 'showermaximum' in prediction:
            if i == 0:
                data['reco_x_max'] = []
            data['reco_x_max'].append(prediction['showermaximum'][0])

        # Store pointing
        if reader.pointing_mode == 'subarray':
            if i == 0:
                data['pointing_alt'], data['pointing_az'] = [],[]
            data['pointing_alt'].append(tel_pointing[0])
            data['pointing_az'].append(tel_pointing[1])

    if prediction_label in list(pd.HDFStore(h5file).keys()):
        pd.HDFStore(h5file).remove(prediction_label)
    pd.DataFrame(data=data).to_hdf(h5file, key=prediction_label, mode='a')
