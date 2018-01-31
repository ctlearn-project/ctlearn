import numpy as np

IMAGE_SHAPES = {
        'MSTS': (120,120,1)
        }

def generate_table_MSTS():
    """
    Function returning MSTS mapping table (used to index into the trace when converting from trace to image).
    """
    
    ROWS = 15
    MODULE_DIM = 8
    MODULES_PER_ROW = [
        5,
        9,
        11,
        13,
        13,
        15,
        15,
        15,
        15,
        15,
        13,
        13,
        11,
        9,
        5]
    
    # bottom left corner of each 8 x 8 module in the camera
    # counting from the bottom row, left to right
    MODULE_START_POSITIONS = [(((IMAGE_SHAPES['MSTS'][0] - MODULES_PER_ROW[j] *
                                 MODULE_DIM) / 2) +
                               (MODULE_DIM * i), j * MODULE_DIM)
                              for j in range(ROWS)
                              for i in range(MODULES_PER_ROW[j])]

    table = np.zeros(shape=(IMAGE_SHAPES['MSTS'][0],IMAGE_SHAPES['MSTS'][1]),dtype=int)   
    # Fill appropriate positions with indices
    # NOTE: we append a 0 entry to the (11328,) trace array to allow us to use fancy indexing to fill
    # the empty areas of the (120,120) image. Accordingly, all indices in the mapping table are increased by 1
    # (j starts at 1 rather than 0)
    j = 1
    for (x_0,y_0) in MODULE_START_POSITIONS:
        for i in range(MODULE_DIM * MODULE_DIM):
            x = int(x_0 + i // MODULE_DIM)
            y = y_0 + i % MODULE_DIM
            table[x][y] = j
            j += 1

    return table

MAPPING_TABLES = {
        'MSTS': generate_table_MSTS()
        }


