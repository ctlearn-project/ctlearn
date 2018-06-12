import numpy as np
import logging
import tables
import threading

from ctalearn.image import MAPPING_TABLES, IMAGE_SHAPES

logger = logging.getLogger(__name__)

# Multithread-safe PyTables open and close file functions
# See http://www.pytables.org/latest/cookbook/threading.html
lock = threading.Lock()

class image_mapper():
    def __init__(self, image_mapping_settings):
        """   
        :param image_mapping_settings: 
        """

        self.IMAGE_SHAPES = {
            'MSTS': (120, 120, 1),
            'VTS': (54, 54, 1),
            #'VTS': (499, 1)
        }

        self.MAPPING_TABLES = {
            'MSTS': self.generate_table_MSTS(),
            'VTS': self.generate_table_VTS()
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

        elif telescope_type == "VTS":
            telescope_image = np.zeros(self.IMAGE_SHAPES['VTS'], dtype=int)
            for i in range(self.IMAGE_SHAPES[telescope_type][0]):
                for j in range(self.IMAGE_SHAPES[telescope_type][1]):
                    telescope_image[i, j, 0] = pixels[self.MAPPING_TABLES[telescope_type][i, j]]

        return telescope_image


    def generate_table_VTS(self):
        """
        Function returning VTS mapping table (used to index into the trace when converting from trace to image).
        Note that for a VERITAS telescope, input trace is of shape (499), while output image is of shape (54, 54, 1)
        """
        # telescope configuration
        NUM_PIXEL = 499
        PIX_SIDE_LENGTH = 1.0 * np.sqrt(2)
        NUM_SPIRAL = 13

        # some internal variables:
        pos = np.zeros((NUM_PIXEL, 2), dtype=float)
        deltaX = PIX_SIDE_LENGTH * np.sqrt(2) / 2.
        deltaY = PIX_SIDE_LENGTH * np.sqrt(2)
        pixNum = 1

        # return mapping_matrix (54, 54)
        # leave 0 for padding, mapping matrix from 1 to 499
        mapping_matrix = np.zeros(self.IMAGE_SHAPES['VTS'][:-1], dtype=int)

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
            mapping_matrix[x_S:x_L + 1, y_S:y_L + 1] = i+1

        return mapping_matrix

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


