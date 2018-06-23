import numpy as np
import logging
#import tables
import threading

from scipy.sparse import csr_matrix


#from ctalearn.image import MAPPING_TABLES, IMAGE_SHAPES

logger = logging.getLogger(__name__)

# Multithread-safe PyTables open and close file functions
# See http://www.pytables.org/latest/cookbook/threading.html
lock = threading.Lock()

class image_mapper():
    def __init__(self, image_mapping_settings=None):
        """   
        :param image_mapping_settings: (Hex converter algorithm, output image shape IMAGE_SHAPES, ...)
        """
        if image_mapping_settings is None: 
            image_mapping_settings = {}
            
        if 'IMAGE_SHAPES' not in image_mapping_settings:
            self.IMAGE_SHAPES = {
                'MSTS': (120, 120, 1),
                'VTS': (54, 54, 1),
                'MSTF': (120, 120, 1),
                'MSTN': (120, 120, 1),
                'LST': (120, 120, 1),
                'SST1': (100, 100, 1),
                'SSTC': (48, 48, 1),
                'SSTA': (56, 56, 1)
                #'VTS': (499, 1)
            }
        else:
            self.IMAGE_SHAPES = image_mapping_settings['IMAGE_SHAPES']

        if 'PIXEL_LENGTH_DICT' not in image_mapping_settings:
            self.PIXEL_LENGTH_DICT = {'LST': 0.05,
                                  'MSTF': 0.05, 'MSTN': 0.05, 'MSTS': 0.00669,
                                  'SST1': 0.0236, 'SSTC':0.0064, 'SSTA':0.0071638,
                                  'VTS': 1.0 / np.sqrt(2)}
        else:
            self.PIXEL_LENGTH_DICT = image_mapping_settings['PIXEL_LENGTH_DICT']

        if 'PIXEL_NUM_DICT' not in image_mapping_settings:
        #self.PIXEL_NUM_DICT = {}
        #########################
        # Have to check!!!
        # Have to check!!!
        # Have to check!!!
            self.PIXEL_NUM_DICT = {'MSTF': 1764, 'MSTN': 1854, 'SST1': 1296,
                                   'LST': 1854, 'MSTS': 11328, 'SSTC': 2048, 'SSTA': 2368,
                                   'VTS': 499}
        #########################
        else:
            self.PIXEL_NUM_DICT = image_mapping_settings['PIXEL_NUM_DICT']
            
        if 'POS_DICT' not in image_mapping_settings:
            self.POS_DICT = {}
        else:
            self.POS_DICT = image_mapping_settings['POS_DICT']

        self.MAPPING_TABLES = {
            'MSTS': self.generate_table_MSTS(),
            'VTS': self.generate_table_VTS(),
            'MSTF': self.generate_table_MSTF(),
            'MSTN': self.generate_table_MSTN(),
            'LST': self.generate_table_LST(),
            'SST1': self.generate_table_SST1(),
            'SSTC': self.generate_table_SSTC(),
            'SSTA': self.generate_table_SSTA()
        }



    def map_image(self, pixels, telescope_type):
        """
        :param pixels: a numpy array of values for each pixel, in order of pixel index.
                       The array has dimensions [N_pixels, N_channels] where N_channels is e.g., 
                       1 when just using charges and 2 when using charges and peak arrival times. 
        :param telescope_type: a string specifying the telescope type as defined in the HDF5 format, 
                        e.g., 'MSTS' for SCT data, which is the only currently implemented telescope type.
        :return: 
        """
        if telescope_type in self.IMAGE_SHAPES.keys():
            self.telescope_type = telescope_type
        else:
            raise ValueError('Sorry! Telescope type {} isn\'t supported.'.format(telescope_type))

        if telescope_type == "MSTS":
            # Create image by indexing into the trace using the mapping table, then adding a
            # dimension to given shape (length,width,1)
            telescope_image = np.expand_dims(pixels[self.MAPPING_TABLES[telescope_type]], 2)


        elif telescope_type in ['LST', 'MSTF', 'MSTN', 'SST1', 'SSTC', 'SSTA', 'VTS']:
            #print("pixels dimension {}".format(pixels.shape))
            #print("MAPPING_TABLES dimension {}".format(self.MAPPING_TABLES[telescope_type].shape))
            telescope_image = (pixels.T @ self.MAPPING_TABLES[telescope_type]).reshape(self.IMAGE_SHAPES[telescope_type][0],
                                                                                       self.IMAGE_SHAPES[telescope_type][1], 1)

        else:
            # should never happen
            telescope_image = None

        return telescope_image


    def generate_table_VTS(self):
        """
        Function returning VTS mapping matrix (used to convert a 1d trace to a resampled 2d image in square lattice).
        Note that for a VERITAS telescope, input trace is of shape (499), while output image is of shape (54, 54, 1)
        The return matrix is of shape (499+1, 54*54) = (500, 2916)
                                      # the added 1 is for the 0th channel = 0 for padding
        To get the image from trace using the return matrix, 
        do: (trace.T @ mapping_matrix3d_sparse).reshape(54,54,1)
        """
        # telescope configuration
        NUM_PIXEL = 499
        PIX_SIDE_LENGTH = 1.0 * np.sqrt(2)
        NUM_SPIRAL = 13

        NORMALIZATION = 1./4. #divide each pixel intensity into 4 sub pixels
        # some internal variables:
        pos = np.zeros((NUM_PIXEL, 2), dtype=float)
        deltaX = PIX_SIDE_LENGTH * np.sqrt(2) / 2.
        deltaY = PIX_SIDE_LENGTH * np.sqrt(2)
        pixNum = 1

        # return mapping_matrix (54, 54)
        # leave 0 for padding, mapping matrix from 1 to 499
        #mapping_matrix = np.zeros(self.IMAGE_SHAPES['VTS'][:-1], dtype=int)
        mapping_matrix = np.zeros((NUM_PIXEL+1, self.IMAGE_SHAPES['VTS'][0], self.IMAGE_SHAPES['VTS'][1]), dtype=float)
        #mapping_matrix3d_sparse = csr_matrix(np.zeros((NUM_PIXEL+1, self.IMAGE_SHAPES['VTS'][0]*self.IMAGE_SHAPES['VTS'][1]), dtype=float))
        #mapping_matrix3d_sparse = csr_matrix(mapping_matrix.reshape(500, 54 * 54))

        for spiral in range(1, NUM_SPIRAL + 1):

            xPos = 2. * float((spiral)) * deltaX
            yPos = 0.

            # For the two outermost spirals, there is not a pixel in the y=0 row.
            if spiral < 12:
                pixNum += 1
                pixNumStr = "{:d}".format(pixNum)
                # plt.text(xPos+self.textOffsetX*(math.floor(math.log10(pixNum)+1.)), yPos+self.textOffsetY, pixNum, size=self.pixLabelFontSize)

                pos[pixNum - 1, 0] = xPos
                pos[pixNum - 1, 1] = yPos

            nextPixDir = np.zeros((spiral * 6, 2))
            skipPixel = np.zeros((spiral * 6, 1))

            for y in range(spiral * 6 - 1):
                # print "%d" % (y/spiral)
                if (y / spiral < 1):
                    nextPixDir[y, :] = [-1, -1]
                elif (y / spiral >= 1 and y / spiral < 2):
                    nextPixDir[y, :] = [-2, 0]
                elif (y / spiral >= 2 and y / spiral < 3):
                    nextPixDir[y, :] = [-1, 1]
                elif (y / spiral >= 3 and y / spiral < 4):
                    nextPixDir[y, :] = [1, 1]
                elif (y / spiral >= 4 and y / spiral < 5):
                    nextPixDir[y, :] = [2, 0]
                elif (y / spiral >= 5 and y / spiral < 6):
                    nextPixDir[y, :] = [1, -1]

            # The two outer spirals are not fully populated with pixels.
            # The second outermost spiral is missing only six pixels (one was excluded above).
            if (spiral == 12):
                for i in range(1, 6):
                    skipPixel[spiral * i - 1] = 1
            # The outmost spiral only has a total of 36 pixels.  We need to skip over the
            # place holders for the rest.
            if (spiral == 13):
                skipPixel[0:3] = 1
                skipPixel[9:16] = 1
                skipPixel[22:29] = 1
                skipPixel[35:42] = 1
                skipPixel[48:55] = 1
                skipPixel[61:68] = 1
                skipPixel[74:77] = 1

            for y in range(spiral * 6 - 1):

                xPos += nextPixDir[y, 0] * deltaX
                yPos += nextPixDir[y, 1] * deltaY

                if skipPixel[y, 0] == 0:
                    pixNum += 1
                    # self.pixNumArr.append(pixNum)
                    pixNumStr = "%d" % pixNum
                    pos[pixNum - 1, 0] = xPos
                    pos[pixNum - 1, 1] = yPos

        pos_shifted = pos + 26 + PIX_SIDE_LENGTH / 2.
        for i in range(NUM_PIXEL):
            x, y = pos_shifted[i, :]
            x_L = int(round(x + PIX_SIDE_LENGTH / 2.))
            x_S = int(round(x - PIX_SIDE_LENGTH / 2.))
            y_L = int(round(y + PIX_SIDE_LENGTH / 2.))
            y_S = int(round(y - PIX_SIDE_LENGTH / 2.))
            # camera.index[0,i,:] = np.array([x_L,x_S,x_S,x_L],dtype=int)
            # camera.index[1,i,:] = np.array([y_L,y_S,y_L,y_S],dtype=int)
            # leave 0 for padding, mapping matrix from 1 to 499
            #mapping_matrix[x_S:x_L + 1, y_S:y_L + 1] = i+1
            mapping_matrix[i + 1, x_S:x_L + 1, y_S:y_L + 1] = NORMALIZATION
            #mapping_matrix3d_sparse[i + 1, x_S:x_L + 1, y_S:y_L + 1] = 1./4.
            #mapping_matrix3d_sparse[i + 1, x_S*self.IMAGE_SHAPES['VTS'][1]+y_S] = 1. / 4.
            #mapping_matrix3d_sparse[i + 1, x_L*self.IMAGE_SHAPES['VTS'][1]+y_S] = 1. / 4.
            #mapping_matrix3d_sparse[i + 1, x_S*self.IMAGE_SHAPES['VTS'][1]+y_S] = 1. / 4.
            #mapping_matrix3d_sparse[i + 1, x_L*self.IMAGE_SHAPES['VTS'][1]+y_L] = 1. / 4.

        #return mapping_matrix
        mapping_matrix3d_sparse = csr_matrix(mapping_matrix.reshape(NUM_PIXEL+1, self.IMAGE_SHAPES['VTS'][0]*self.IMAGE_SHAPES['VTS'][1]))
        return mapping_matrix3d_sparse


    def generate_table_MSTS(self):
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
        MODULE_START_POSITIONS = [(((self.IMAGE_SHAPES['MSTS'][0] - MODULES_PER_ROW[j] *
                                     MODULE_DIM) / 2) +
                                   (MODULE_DIM * i), j * MODULE_DIM)
                                  for j in range(ROWS)
                                  for i in range(MODULES_PER_ROW[j])]

        table = np.zeros(shape=(self.IMAGE_SHAPES['MSTS'][0], self.IMAGE_SHAPES['MSTS'][1]), dtype=int)
        # Fill appropriate positions with indices
        # NOTE: we append a 0 entry to the (11328,) trace array to allow us to use fancy indexing to fill
        # the empty areas of the (120,120) image. Accordingly, all indices in the mapping table are increased by 1
        # (j starts at 1 rather than 0)
        j = 1
        for (x_0, y_0) in MODULE_START_POSITIONS:
            for i in range(MODULE_DIM * MODULE_DIM):
                x = int(x_0 + i // MODULE_DIM)
                y = y_0 + i % MODULE_DIM
                table[x][y] = j
                j += 1

        return table


    def generate_table_SSTC(self):
        """
        Function returning SSTC mapping table (used to index into the trace when converting from trace to image).
        """
        this_tel = "SSTC"
        MODULES_PER_ROW_DICT = { 0: 32,
                                 1: 32,
                                 2: 32,
                                 3: 32,
                                 4: 32,
                                 5: 32,
                                 6: 32,
                                 7: 32,
                                 8: 48,
                                 9: 48,
                                 10: 48,
                                 11: 48,
                                 12: 48,
                                 13: 48,
                                 14: 48,
                                 15: 48,
                                 16: 48,
                                 17: 48,
                                 18: 48,
                                 19: 48,
                                 20: 48,
                                 21: 48,
                                 22: 48,
                                 23: 48,
                                 24: 48,
                                 25: 48,
                                 26: 48,
                                 27: 48,
                                 28: 48,
                                 29: 48,
                                 30: 48,
                                 31: 48,
                                 32: 48,
                                 33: 48,
                                 34: 48,
                                 35: 48,
                                 36: 48,
                                 37: 48,
                                 38: 48,
                                 39: 48,
                                 40: 32,
                                 41: 32,
                                 42: 32,
                                 43: 32,
                                 44: 32,
                                 45: 32,
                                 46: 32,
                                 47: 32 }

        # This is set to int because no oversampling is done
        mapping_matrix3d = np.zeros((self.PIXEL_NUM_DICT[this_tel] + 1,
                                     self.IMAGE_SHAPES[this_tel][0],
                                     self.IMAGE_SHAPES[this_tel][1]), dtype=int)

        i = 0  # Pixel count
        for row_, n_per_row_ in MODULES_PER_ROW_DICT.items():
            row_start_ = (self.IMAGE_SHAPES[this_tel][1] - n_per_row_) / 2
            for j in range(n_per_row_):
                x, y = (row_, j + row_start_)
                # print(x,y)
                mapping_matrix3d[i + 1, x, y] = 1
                i = i + 1

        sparse_map_mat = csr_matrix(mapping_matrix3d.reshape(self.PIXEL_NUM_DICT[this_tel] + 1,
                                                             self.IMAGE_SHAPES[this_tel][0]*
                                                             self.IMAGE_SHAPES[this_tel][1]))

        return sparse_map_mat


    def generate_table_SSTA(self):
        """
        Function returning SSTC mapping table (used to index into the trace when converting from trace to image).
        """
        this_tel = "SSTA"
        MODULES_PER_ROW_DICT = { 0: 24,
                                 1: 24,
                                 2: 24,
                                 3: 24,
                                 4: 24,
                                 5: 24,
                                 6: 24,
                                 7: 24,
                                 8: 40,
                                 9: 40,
                                 10: 40,
                                 11: 40,
                                 12: 40,
                                 13: 40,
                                 14: 40,
                                 15: 40,
                                 16: 56,
                                 17: 56,
                                 18: 56,
                                 19: 56,
                                 20: 56,
                                 21: 56,
                                 22: 56,
                                 23: 56,
                                 24: 56,
                                 25: 56,
                                 26: 56,
                                 27: 56,
                                 28: 56,
                                 29: 56,
                                 30: 56,
                                 31: 56,
                                 32: 56,
                                 33: 56,
                                 34: 56,
                                 35: 56,
                                 36: 56,
                                 37: 56,
                                 38: 56,
                                 39: 56,
                                 40: 40,
                                 41: 40,
                                 42: 40,
                                 43: 40,
                                 44: 40,
                                 45: 40,
                                 46: 40,
                                 47: 40,
                                 48: 24,
                                 49: 24,
                                 50: 24,
                                 51: 24,
                                 52: 24,
                                 53: 24,
                                 54: 24,
                                 55: 24}
        # This is set to int because no oversampling is done
        mapping_matrix3d = np.zeros((self.PIXEL_NUM_DICT[this_tel] + 1,
                                     self.IMAGE_SHAPES[this_tel][0],
                                     self.IMAGE_SHAPES[this_tel][1]), dtype=int)

        i = 0  # Pixel count
        for row_, n_per_row_ in MODULES_PER_ROW_DICT.items():
            row_start_ = (self.IMAGE_SHAPES[this_tel][1] - n_per_row_) / 2
            for j in range(n_per_row_):
                x, y = (row_, j + row_start_)
                # print(x,y)
                mapping_matrix3d[i + 1, x, y] = 1
                i = i + 1

        sparse_map_mat = csr_matrix(mapping_matrix3d.reshape(self.PIXEL_NUM_DICT[this_tel] + 1,
                                                             self.IMAGE_SHAPES[this_tel][0]*
                                                             self.IMAGE_SHAPES[this_tel][1]))

        return sparse_map_mat


    def get_mat_from_pos(self, pos, numpix, dim1=120, dim2=120, pixel_length=0.05, pixel_weight=1. / 4):
        """
        :param pos: (2,numpix)
        :param numpix: 
        :param dim2d: for output square array size
        :param pixel_length: distance between the center of two pix
        :param pixel_weight: e.g., 1 pixel is oversampled into 4, so weigh pix values
        :return: 
        """
        pos_int = pos / pixel_length * 2 + 1
        pos_int[0, :] = pos_int[0, :] / np.sqrt(3) * 2
        pos_int[0, :] -= np.min(pos_int[0, :])
        pos_int[1, :] -= np.min(pos_int[1, :])

        mapping_matrix3d = np.zeros((numpix + 1, dim1, dim2), dtype=float)

        for i in range(numpix):
            x, y = pos_int[:, i]
            x_S = int(round(x))
            x_L = x_S + 1
            y_S = int(round(y))
            y_L = y_S + 1
            # leave 0 for padding, mapping matrix from 1 to 499
            #if i < 10:
            #    print(x_S, x_L + 1, y_S, y_L + 1)
            #mapping_matrix3d[i + 1, x_S:x_L + 1, y_S:y_L + 1] = pixel_weight
            #########################
            # Have to check!!!
            # Have to check!!!
            # Have to check!!!
            # Not sure why x and y are flipped, maybe because of numpy storage row/col first rule?
            mapping_matrix3d[i + 1, y_S:y_L + 1, x_S:x_L + 1] = pixel_weight

        sparse_map_mat = csr_matrix(mapping_matrix3d.reshape(numpix + 1, dim1 * dim2))

        return sparse_map_mat


    def generate_table_generic(self, this_tel, pixel_weight=1./4):
        # only for hex cameras
        # some internal variables:
        if this_tel not in self.POS_DICT:
            pos = read_pix_pos_files(this_tel)
        else:
            pos = self.POS_DICT[this_tel]

        if this_tel in ["LST", "MSTN"]:
            pos = self.rot_lst(pos)

        if this_tel not in self.IMAGE_SHAPES:
            dim1 = 120 #guess 120
            dim2 = 120
        else:
            dim1 = self.IMAGE_SHAPES[this_tel][0]
            dim2 = self.IMAGE_SHAPES[this_tel][1]

        if this_tel not in self.PIXEL_NUM_DICT:
            pos_sq_ = pos[0, :] ** 2 + pos[1, :] ** 2
            numpix = pos_sq_[np.where(pos_sq_>0)].shape[0]
        else:
            numpix = self.PIXEL_NUM_DICT[this_tel]

        if this_tel not in self.PIXEL_LENGTH_DICT:
            #better not happen...
            pixel_length_ = 0.05 # guess
        else:
            pixel_length_ = self.PIXEL_LENGTH_DICT[this_tel]

        return self.get_mat_from_pos(pos, numpix, dim1 = dim1, dim2=dim2,
                                     pixel_length=pixel_length_, pixel_weight=pixel_weight)



    def generate_table_MSTF(self):
        """
        Function returning MSTF mapping matrix (used to convert a 1d trace to a resampled 2d image in square lattice).
        input is a 1-d vecotr of shape (1765, )
        """
        #NUM_PIXEL = 1764
        return self.generate_table_generic("MSTF")


    def generate_table_MSTN(self):

        return self.generate_table_generic("MSTN")


    def generate_table_LST(self):

        return self.generate_table_generic("LST")


    def generate_table_SST1(self):

        return self.generate_table_generic("SST1")

    def rot_lst(self, pos):
        # rotating LST/MSTN camera!!!
        # c, s = np.cos(-np.arctan((POS_DICT['LST'][0,0] - POS_DICT['LST'][0,1]) /
        #                            (POS_DICT['LST'][1,0] - POS_DICT['LST'][1,1]) ) ), \
        #       np.sin(-np.arctan((POS_DICT['LST'][0,0] - POS_DICT['LST'][0,1]) /
        #                            (POS_DICT['LST'][1,0] - POS_DICT['LST'][1,1]) ) )
        # rot_mat = np.matrix([[c, s], [-s, c]])
        rot_mat = np.matrix([[0.98198181, 0.18897548],
                             [-0.18897548, 0.98198181]], dtype=float)

        lst_pos_rot = np.squeeze(np.asarray(np.dot(rot_mat, pos)))

        return lst_pos_rot


    def rebinning(self):
        #placeholder

        return




# utility functions to create pixel pos numpy files:

def check_tel(tel):
    if tel not in ['MSTF', 'MSTS', 'MSTN', 'LST', 'SST1']:
        print("Telescope type {} isn't supported.".format(tel))
        return -1
    return 1

def get_pos_from_h5(telTab, tel="MSTF", write=False, outfile=None):
    ind_ = np.array([row.nrow for row in telTab.where('tel_type==tel')])[0]
    if not write:
        return telTab.cols.pixel_pos[ind_]
    else:
        pos_ = telTab.cols.pixel_pos[ind_]
        if outfile is None:
            outfile = "{}_pos".format(tel)
        np.save(outfile, pos_)
        return pos_

def create_pix_pos_files(h5file):
    import tables # expect this to be run very rarely...

    h1=tables.open_file(h5file, "r")
    telTab=h1.root.Telescope_Info
    for row_ in telTab.iterrows():
        #print((row_[1]).decode("utf-8"))
        _ = get_pos_from_h5(telTab, tel=row_[1].decode("utf-8"), write=True)
    h1.close()

def read_pix_pos_files(tel):
    if not check_tel(tel):
        return -1
    infile = "{}_pos.npy".format(tel)
    return np.load(infile)