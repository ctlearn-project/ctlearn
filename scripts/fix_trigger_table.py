"""
When running a mono analysis on stereo MC (i.e., selecting only 1 telescope in 
train and prediction configuration file), dl2 output data only has events triggered
by the desired telescope in dl2, but in dl1 trigger table all the events (also the ones
not triggered by the desired telescope) are present, producing an error when ctapipe 
reads the files to produce IRFs. This script can be used to 'cut' the trigger table and 
solve this issue
"""
import os
import glob
from astropy.table import Table
import tables
import pandas as pd
import  numpy as np
from astropy.io.misc.hdf5 import write_table_hdf5

def main():
    os.system('cp -r ./DL2 ./DL2_original')
    #create a copy of all the original files before modifying anything
    particle=os.listdir('./DL2/MC/LST_mono_4LST')
    for i in particle:
        filelist=glob.glob(f'./DL2/MC/LST_mono_4LST/{i}/20.000_180.000/*.h5')
        for file in filelist:
            input_file=tables.open(file)
            df = Table(input_file.root.dl1.event.subarray.trigger[:])
            trigger=pd.Dataframe()
            trigger['obs_id']=df['obs_id']
            trigger['event_id']=df['event_id']
            df2 = Table(input_file.root.dl2.event.subarray.energy.CTLearn[:])
            energy=pd.DataFrame()
            energy['obs_id']=df2['obs_id']
            energy['event_id']=df2['event_id']
            trigger=trigger.groupby('obs_id')
            mask=[]
            for name, group in trigger:
                energypar=energy[energy['obs_id']==name]
                maskpar=group['event_id'].isin(energypar['event_id'])
                mask.append(maskpar)
            mask=pd.concat(mask)
            dfcut=df[mask]
            write_table_hdf5(
                dfcut, 
                file , 
                path='/dl1/event/subarray/trigger', 
                overwrite=True, 
                append=True
            )





if __name__ == "__main__":
    main()