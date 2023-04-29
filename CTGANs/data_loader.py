from dl1_data_handler.reader import DL1DataReaderSTAGE1, DL1DataReaderDL1DH
from ctlearn.data_loader import KerasBatchGenerator
from ctlearn.utils import *


# TODO: scale images to [-1, 1] range
def load_data(config, batch_size=64, shuffle=False, mode='train'):
    # Set up the DL1DataReader
    config['Data'], data_format = setup_DL1DataReader(config, mode)
    # Create data reader
    if data_format == 'stage1':
        reader = DL1DataReaderSTAGE1(**config['Data'])
    elif data_format == 'dl1dh':
        reader = DL1DataReaderDL1DH(**config['Data'])

    # Set up the KerasBatchGenerator
    indices = list(range(len(reader)))
    dataset = KerasBatchGenerator(reader, indices, batch_size=batch_size, mode=mode, shuffle=shuffle)

    return dataset