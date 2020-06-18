import logging
import numpy as np
import pandas as pd
import ctaplot
from sklearn import metrics

def write_output(h5file, data, predictions, step=0, prediction_label='prediction'):
    for i, prediction in enumerate(predictions):
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
            if data['energy_unit'] == 'log(TeV)':
                prediction['energy'][0] = np.power(10,prediction['energy'][0])
            data['reco_energy'].append(prediction['energy'][0])
        # Arrival direction regression
        if 'direction' in prediction:
            if i == 0:
                data['reco_altitude'], data['reco_azimuth'] = [],[]
            data['reco_altitude'].append(prediction['direction'][0] + data['tel_pointing'][1])
            data['reco_azimuth'].append(prediction['direction'][1] + data['tel_pointing'][0])
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
    
    if step > 0:
        calculate_validation_step(h5file, data, step)
    if step == 0:
        if 'final_step' in list(pd.HDFStore(h5file).keys()):
            pd.HDFStore(h5file).remove('final_step')
        del data['energy_unit']
        del data['tel_pointing']
        pd.DataFrame(data=data).to_hdf(h5file, key='final_step', mode='a')
    if step == -1:
        if prediction_label in list(pd.HDFStore(h5file).keys()):
            pd.HDFStore(h5file).remove(prediction_label)
        del data['energy_unit']
        del data['tel_pointing']
        pd.DataFrame(data=data).to_hdf(h5file, key=prediction_label, mode='a')
    return

def calculate_validation_step(h5file, data, step):
    resolution = {}
    if 'reco_gammaness' in data and 'reco_particle' in data:
        # TODO multiclass handling
        # ROC curve
        roc_curve, particle = {}, {}
        mc_particle = np.array(data['mc_particle']).astype(int)
        reco_particle = np.array(data['reco_particle']).astype(int)
        reco_gammaness = np.array(data['reco_gammaness']).astype(float)
        roc_curve['fpr'], roc_curve['tpr'], thresholds = metrics.roc_curve(mc_particle, reco_gammaness)
        pd.DataFrame(data=roc_curve).to_hdf(h5file, key='validation_step{}/ROC_curve/'.format(step), mode='a')
        # Accuracy and AUC
        metric = [metrics.accuracy_score(mc_particle, reco_particle)]
        metric.append(metrics.auc(roc_curve['fpr'], roc_curve['tpr']))
        mask_gamma = (mc_particle == 1)
        mask_proton = (mc_particle == 0)
        metric.append(metrics.accuracy_score(mc_particle[mask_gamma], reco_particle[mask_gamma]))
        metric.append(metrics.accuracy_score(mc_particle[mask_proton], reco_particle[mask_proton]))
        particle['metric_tensor'] = metric
        pd.DataFrame(data=particle).to_hdf(h5file, key='validation_step{}/Particle_classification/'.format(step), mode='a')

    if 'reco_energy' in data:
        mc_energy = np.array(data['mc_energy'])
        reco_energy = np.array(data['reco_energy'])
        if 'mc_particle' in data:
            mc_particle = np.array(data['mc_particle']).astype(int)
            mask_gamma = (mc_particle == 1)
            mc_energy = mc_energy[mask_gamma]
            reco_energy = reco_energy[mask_gamma]
        # Energy resolution
        ebin, eres = ctaplot.ana.energy_resolution_per_energy(mc_energy, reco_energy)
        resolution['Energy_resolution'] = eres[:,0]
        resolution['Energy_resolution_lower'] = eres[:,1]
        resolution['Energy_resolution_upper'] = eres[:,2]
        # Energy bias
        ebin, resolution['Energy_bias'] = ctaplot.ana.energy_bias(mc_energy, reco_energy)

    if 'reco_altitude' in data and 'reco_azimuth' in data:
        
        # Angular resolution
        mc_energy = np.array(data['mc_energy'])
        reco_altitude = np.array(data['reco_altitude']).astype(float)
        reco_azimuth = np.array(data['reco_azimuth']).astype(float)
        mc_altitude = np.array(data['mc_altitude']).astype(float)
        mc_azimuth = np.array(data['mc_azimuth']).astype(float)
        
        if 'mc_particle' in data:
            mc_energy = mc_energy[mask_gamma]
            reco_altitude = reco_altitude[mask_gamma]
            reco_azimuth = reco_azimuth[mask_gamma]
            mc_altitude = mc_altitude[mask_gamma]
            mc_azimuth = mc_azimuth[mask_gamma]

        ebin, angres = ctaplot.ana.angular_resolution_per_energy(reco_altitude, reco_azimuth, mc_altitude, mc_azimuth, mc_energy)
        
        resolution['Angular_resolution'] = angres[:,0]
        resolution['Angular_resolution_lower'] = angres[:,1]
        resolution['Angular_resolution_upper'] = angres[:,2]

    if resolution:
        pd.DataFrame(data=resolution).to_hdf(h5file, key='validation_step{}/Resolution/'.format(step), mode='a')
        
    return
