import numpy as np
import logging
import threading
import os
import cv2

from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# Multithread-safe PyTables open and close file functions
# See http://www.pytables.org/latest/cookbook/threading.html
lock = threading.Lock()

# TEMPORARY: Maps old telescope names into new
# For training with datasets generated with versions prior to image-extractor v0.6.0
OLD_TEL_NAMES_TO_NEW = {
    'LST':'LST:LSTCam',
    'MSTF':'MST:FlashCam',
    'MSTN':'MST:NectarCam',
    'MSTS':'SCT:SCTCam',
    'SSTC':'SST:CHEC',
    'SST1':'SST:DigiCam',
    'SSTA':'SST:ASTRICam'
}

NEW_TEL_NAMES_TO_OLD = {v: k for k, v in OLD_TEL_NAMES_TO_NEW.items()}

class ImageMapper():

    pixel_lengths = {
        'LSTCam': 0.05,
        'FlashCam': 0.05,
        'NectarCam': 0.05,
        'SCTCam': 0.00669,
        'DigiCam': 0.0236,
        'CHEC':0.0064,
        'ASTRICam':0.0071638,
        'VERITAS': 1.0 * np.sqrt(2),
        'MAGICCam': 1.0 * np.sqrt(2),
        'FACT': 0.0095,
        'HESS-I': 0.0514,
        'HESS-II': 0.0514
        }

    num_pixels = {
        'LSTCam': 1855,
        'FlashCam': 1764,
        'NectarCam': 1855,
        'SCTCam': 11328,
        'DigiCam': 1296,
        'CHEC': 2048,
        'ASTRICam': 2368,
        'VERITAS': 499,
        'MAGICCam': 1039,
        'FACT': 1440,
        'HESS-I': 960,
        'HESS-II': 2048
        }

    def __init__(self,
                 hex_conversion_algorithm='oversampling',
                 padding=None,
                 use_peak_times=False):
        """
        :param hex_conversion_algorithm: algorithm to be used when converting
                                         hexagonal pixel camera data to square images
        :param padding: number of pixels of padding to be added symmetrically to the sides
                        of the square image
        :param use_peak_times: if true, the number of input channels is 2
                               (charges and peak arrival times)
        """

        # image_shapes should be a non static field to prevent problems
        # when multiple instances of ImageMapper are created
        self.image_shapes = {
            'LSTCam': (108, 108, 1),
            'FlashCam': (110, 110, 1),
            'NectarCam': (108, 108, 1),
            'SCTCam': (120, 120, 1),
            'DigiCam': (94, 94, 1),
            'CHEC': (48, 48, 1),
            'ASTRICam': (56, 56, 1),
            'VERITAS': (54, 54, 1),
            'MAGICCam': (82, 82, 1),
            'FACT': (90,90,1),
            'HESS-I': (72,72,1),
            'HESS-II': (104,104,1)
            }

        if hex_conversion_algorithm in ['oversampling']:
            self.hex_conversion_algorithm = hex_conversion_algorithm
        else:
            raise NotImplementedError("Hex conversion algorithm {} is not implemented.".format(hex_conversion_algorithm))

        if padding is None:
            padding = {
                    'LSTCam': 0,
                    'FlashCam': 0,
                    'NectarCam': 0,
                    'SCTCam': 0,
                    'DigiCam': 0,
                    'CHEC': 0,
                    'ASTRICam': 0,
                    'VERITAS': 0,
                    'MAGICCam': 0,
                    'FACT': 0,
                    'HESS-I': 0,
                    'HESS-II': 0
                    }
        self.padding = padding

        for tel_, pad_ in self.padding.items():
            if pad_ > 0:
                self.image_shapes[tel_] = (
                    self.image_shapes[tel_][0] + pad_ * 2,
                    self.image_shapes[tel_][1] + pad_ * 2,
                    self.image_shapes[tel_][2]
                )

        if use_peak_times:
            for tel_ in self.image_shapes:
                self.image_shapes[tel_] = (
                    self.image_shapes[tel_][0],
                    self.image_shapes[tel_][1],
                    2 # number of channels
                )

        self.pixel_positions = {camera_type:self.__read_pix_pos_files(camera_type) for camera_type in self.pixel_lengths if (camera_type != 'VERITAS' and camera_type != 'MAGICCam')}

        self.mapping_tables = {
            'LSTCam': self.generate_table_generic('LSTCam'),
            'FlashCam': self.generate_table_generic('FlashCam'),
            'NectarCam': self.generate_table_generic('NectarCam'),
            'SCTCam': self.generate_table_SCTCam(),
            'DigiCam': self.generate_table_generic('DigiCam'),
            'CHEC': self.generate_table_CHEC(),
            'ASTRICam': self.generate_table_ASTRICam(),
            'VERITAS': self.generate_table_VERITAS(),
            'MAGICCam': self.generate_table_MAGICCam(),
            'FACT': self.generate_table_generic('FACT'),
            'HESS-I': self.generate_table_HESS('HESS-I'),
            'HESS-II': self.generate_table_HESS('HESS-II')
            }

    def map_image(self, pixels, camera_type):
        """
        :param pixels: a numpy array of values for each pixel, in order of pixel index.
                       For future reference:
                       The array should have dimensions [N_pixels, N_channels] where N_channels is e.g.,
                       1 when just using charges and 2 when using charges and peak arrival times.
        :param camera_type: a string specifying the camera type as defined in the HDF5 format,
                        e.g., 'SCTCam' for SCT data.
        :return: a numpy array of shape [img_width, img_length, N_channels]
        """
        if camera_type in self.image_shapes.keys():
            self.camera_type = camera_type
        else:
            raise ValueError('Sorry! Camera type {} isn\'t supported.'.format(camera_type))

        n_channels = pixels.shape[1]

        # We reshape each channel and then stack the result
        result = []
        for channel in range(n_channels):
            vector = pixels[:,channel]

            if camera_type == "SCTCam":
                image_2D = vector[self.mapping_tables[camera_type]].T[:,:,np.newaxis]
            else:
                image_2D = (vector.T @ self.mapping_tables[camera_type]).reshape(self.image_shapes[camera_type][0],
                                                                                       self.image_shapes[camera_type][1], 1)
            result.append(image_2D)

        telescope_image = np.concatenate(result, axis = -1)

        if camera_type == "MAGICCam":
            pad = self.padding[camera_type]
            (h, w) = telescope_image.shape[:2]
            h+=pad*2
            w+=pad*2
            center = (w/2.0, h/2.0)
            angle=-19.1
            scale=1.0
            M = cv2.getRotationMatrix2D(center, angle, scale)
            telescope_image = cv2.warpAffine(telescope_image, M, (w, h))
            telescope_image = np.expand_dims(telescope_image, axis=2)

        return telescope_image


    def generate_table_VERITAS(self):
        """
        Function returning VERITAS mapping matrix (used to convert a 1d trace to a resampled 2d image in square lattice).
        Note that for a VERITAS telescope, input trace is of shape (499), while output image is of shape (54, 54, 1)
        The return matrix is of shape (499+1, 54*54) = (500, 2916)
                                      # the added 1 is for the 0th channel = 0 for padding
        To get the image from trace using the return matrix,
        do: (trace.T @ mapping_matrix3d_sparse).reshape(54,54,1)
        """
        # telescope hardcoded values
        num_pixels = 499
        pixel_side_len = self.pixel_lengths['VERITAS']
        num_spirals = 13

        pixel_weight = 1.0/4 #divide each pixel intensity into 4 sub pixels

        pos = np.zeros((num_pixels, 2), dtype=float)
        delta_x = pixel_side_len * np.sqrt(2) / 2.
        delta_y = pixel_side_len * np.sqrt(2)

        pixel_index = 1

        # return mapping_matrix (54, 54)
        # leave 0 for padding, mapping matrix from 1 to 499
        mapping_matrix = np.zeros((num_pixels + 1, self.image_shapes['VERITAS'][0], self.image_shapes['VERITAS'][1]), dtype=float)

        for i in range(1, num_spirals + 1):
            x = 2.0 * i * delta_x
            y = 0.0

            # For the two outermost spirals, there is not a pixel in the y=0 row.
            if i < 12:
                pixel_index += 1

                pos[pixel_index - 1, 0] = x
                pos[pixel_index - 1, 1] = y

            next_pix_dir = np.zeros((i * 6, 2))
            skip_pixel = np.zeros((i * 6, 1))

            for j in range(i * 6 - 1):
                if (j / i < 1):
                    next_pix_dir[j, :] = [-1, -1]
                elif (j / i >= 1 and j / i < 2):
                    next_pix_dir[j, :] = [-2, 0]
                elif (j / i >= 2 and j / i < 3):
                    next_pix_dir[j, :] = [-1, 1]
                elif (j / i >= 3 and j / i < 4):
                    next_pix_dir[j, :] = [1, 1]
                elif (j / i >= 4 and j / i < 5):
                    next_pix_dir[j, :] = [2, 0]
                elif (j / i >= 5 and j / i < 6):
                    next_pix_dir[j, :] = [1, -1]

            # The two outer spirals are not fully populated with pixels.
            # The second outermost spiral is missing only six pixels (one was excluded above).
            if (i == 12):
                for k in range(1, 6):
                    skip_pixel[i * k - 1] = 1
            # The outmost spiral only has a total of 36 pixels.  We need to skip over the
            # place holders for the rest.
            if (i == 13):
                skip_pixel[0:3] = 1
                skip_pixel[9:16] = 1
                skip_pixel[22:29] = 1
                skip_pixel[35:42] = 1
                skip_pixel[48:55] = 1
                skip_pixel[61:68] = 1
                skip_pixel[74:77] = 1

            for j in range(i * 6 - 1):

                x += next_pix_dir[j, 0] * delta_x
                y += next_pix_dir[j, 1] * delta_y

                if skip_pixel[j, 0] == 0:
                    pixel_index += 1
                    pos[pixel_index - 1, 0] = x
                    pos[pixel_index - 1, 1] = y

        pos_shifted = pos + 26 + pixel_side_len / 2.0
        delta_x = int((self.image_shapes['VERITAS'][0] - 54) / 2)
        delta_y = int((self.image_shapes['VERITAS'][1] - 54) / 2)

        for i in range(num_pixels):
            x, y = pos_shifted[i, :]
            x_L = int(round(x + pixel_side_len / 2.0)) + delta_x
            x_S = int(round(x - pixel_side_len / 2.0)) + delta_x
            y_L = int(round(y + pixel_side_len / 2.0)) + delta_y
            y_S = int(round(y - pixel_side_len / 2.0)) + delta_y

            # leave 0 for padding, mapping matrix from 1 to 499
            #mapping_matrix[i + 1, x_S:x_L + 1, y_S:y_L + 1] = pixel_weight
            mapping_matrix[i + 1, y_S:y_L + 1, x_S:x_L + 1] = pixel_weight

        mapping_matrix = csr_matrix(mapping_matrix.reshape(num_pixels + 1, self.image_shapes['VERITAS'][0] * self.image_shapes['VERITAS'][1]))

        return mapping_matrix

    def generate_table_MAGICCam(self):
        """
            Function returning MAGICCam mapping matrix (used to convert a 1d trace to a resampled 2d image in square lattice).
            Note that for a MAGIC telescope, input trace is of shape (1039), while output image is of shape (82, 82, 1)
            The return matrix is of shape (1039+1, 82*82) = (1040, 6724)
            # the added 1 is for the 0th channel = 0 for padding
            To get the image from trace using the return matrix,
            do: (trace.T @ mapping_matrix3d_sparse).reshape(82,82,1)
            """
        # telescope hardcoded values
        num_pixels = 1039
        pixel_side_len = self.pixel_lengths['MAGICCam']
        num_spirals = 19

        pixel_weight = 1.0/4 #divide each pixel intensity into 4 sub pixels

        pos = np.zeros((num_pixels, 2), dtype=float)
        delta_x = pixel_side_len * np.sqrt(2) / 2.
        delta_y = pixel_side_len * np.sqrt(2)

        pixel_index = 1

        # return mapping_matrix (82, 82)
        # leave 0 for padding, mapping matrix from 1 to 1038
        mapping_matrix = np.zeros((num_pixels + 1, self.image_shapes['MAGICCam'][0], self.image_shapes['MAGICCam'][1]), dtype=float)

        for i in range(1, num_spirals + 1):

            x = 2.0 * i * delta_x
            y = 0.0

            # For the three outermost spirals, there is not a pixel in the y=0 row.
            if i < 17:
                pixel_index += 1

                pos[pixel_index - 1, 0] = x
                pos[pixel_index - 1, 1] = y

            next_pix_dir = np.zeros((i * 6, 2))
            skip_pixel = np.zeros((i * 6, 1))
            for j in range(i * 6 - 1):
                if (j / i < 1):
                    next_pix_dir[j, :] = [-1, 1]
                elif (j / i >= 1 and j / i < 2):
                    next_pix_dir[j, :] = [-2, 0]
                elif (j / i >= 2 and j / i < 3):
                    next_pix_dir[j, :] = [-1, -1]
                elif (j / i >= 3 and j / i < 4):
                    next_pix_dir[j, :] = [1, -1]
                elif (j / i >= 4 and j / i < 5):
                    next_pix_dir[j, :] = [2, 0]
                elif (j / i >= 5 and j / i < 6):
                    next_pix_dir[j, :] = [1, 1]

            # The three outer spirals are not fully populated with pixels.
            # The third outermost spiral is missing only six pixels (one was excluded above).
            if (i == 17):
                for k in range(1, 6):
                    skip_pixel[i * k - 1] = 1
            # The second outermost spiral only has a total of 78 pixels.  We need to skip over the
            # place holders for the rest.
            if (i == 18):
                skip_pixel[0:2] = 1
                skip_pixel[15:20] = 1
                skip_pixel[33:38] = 1
                skip_pixel[51:56] = 1
                skip_pixel[69:74] = 1
                skip_pixel[87:92] = 1
                skip_pixel[105:107] = 1

            # The outmost spiral only has a total of 48 pixels.  We need to skip over the
            # place holders for the rest.
            if (i == 19):
                skip_pixel[0:5] = 1
                skip_pixel[13:24] = 1
                skip_pixel[32:43] = 1
                skip_pixel[51:62] = 1
                skip_pixel[70:81] = 1
                skip_pixel[89:100] = 1
                skip_pixel[108:113] = 1

            for j in range(i * 6 - 1):
                x += next_pix_dir[j, 0] * delta_x
                y += next_pix_dir[j, 1] * delta_y

                if skip_pixel[j, 0] == 0:
                    pixel_index += 1
                    pos[pixel_index - 1, 0] = x
                    pos[pixel_index - 1, 1] = y

        pos_shifted = pos + (self.image_shapes['MAGICCam'][0]-2)/2 + pixel_side_len / 2.0

        delta_x = int((self.image_shapes['MAGICCam'][0] - 82) / 2)
        delta_y = int((self.image_shapes['MAGICCam'][1] - 82) / 2)

        for i in range(num_pixels):
            x, y = pos_shifted[i, :]
            x_L = int(round(x + pixel_side_len / 2.0)) + delta_x
            x_S = int(round(x - pixel_side_len / 2.0)) + delta_x
            y_L = int(round(y + pixel_side_len / 2.0)) + delta_y
            y_S = int(round(y - pixel_side_len / 2.0)) + delta_y

            # leave 0 for padding, mapping matrix from 1 to 1038
            #mapping_matrix[i + 1, x_S:x_L + 1, y_S:y_L + 1] = pixel_weight
            mapping_matrix[i + 1, y_S:y_L + 1, x_S:x_L + 1] = pixel_weight

        mapping_matrix = csr_matrix(mapping_matrix.reshape(num_pixels + 1, self.image_shapes['MAGICCam'][0] * self.image_shapes['MAGICCam'][1]))

        return mapping_matrix


    def generate_table_SCTCam(self):
        """
        Function returning SCTCam mapping table (used to index into the trace when converting from trace to image).
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
        MODULE_START_POSITIONS = [(((self.image_shapes['SCTCam'][0] - MODULES_PER_ROW[j] *
                                     MODULE_DIM) / 2) +
                                   (MODULE_DIM * i),
                                   ((self.image_shapes['SCTCam'][1] - MODULES_PER_ROW[ROWS//2] *
                                     MODULE_DIM) / 2) +
                                   (j * MODULE_DIM))
                                  for j in range(ROWS)
                                  for i in range(MODULES_PER_ROW[j])]

        table = np.zeros(shape=(self.image_shapes['SCTCam'][0], self.image_shapes['SCTCam'][1]), dtype=int)
        # Fill appropriate positions with indices
        # NOTE: we append a 0 entry to the (11328,) trace array to allow us to use fancy indexing to fill
        # the empty areas of the (120,120) image. Accordingly, all indices in the mapping table are increased by 1
        # (j starts at 1 rather than 0)
        j = 1
        for (x_0, y_0) in MODULE_START_POSITIONS:
            for i in range(MODULE_DIM * MODULE_DIM):
                x = int(x_0 + i // MODULE_DIM)
                y = int(y_0 + i % MODULE_DIM)
                table[x][y] = j
                j += 1

        return table

    def generate_table_CHEC(self):
        """
        Function returning CHEC mapping table (used to index into the trace when converting from trace to image).
        """

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
        mapping_matrix3d = np.zeros((self.num_pixels['CHEC'] + 1,
                                     self.image_shapes['CHEC'][0],
                                     self.image_shapes['CHEC'][1]), dtype=int)

        i = 0  # Pixel count
        # offset vertically:
        delta_x = int((self.image_shapes['CHEC'][1] - MODULES_PER_ROW_DICT[23]) / 2)
        for row_, n_per_row_ in MODULES_PER_ROW_DICT.items():
            row_start_ = int((self.image_shapes['CHEC'][0] - n_per_row_) / 2)
            for j in range(n_per_row_):
                x, y = (row_ + delta_x, j + row_start_)
                mapping_matrix3d[i + 1, x, y] = 1
                i += 1

        sparse_map_mat = csr_matrix(mapping_matrix3d.reshape(self.num_pixels['CHEC'] + 1,
                                                             self.image_shapes['CHEC'][0]*
                                                             self.image_shapes['CHEC'][1]))

        return sparse_map_mat


    def generate_table_ASTRICam(self):
        """
        Function returning ASTRICam mapping table (used to index into the trace when converting from trace to image).
        """
        img_map = np.full([56, 56], -1, dtype=int)

        # Map values
        img_map[0:8, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 34 * 64
        img_map[0:8, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 35 * 64
        img_map[0:8, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 36 * 64

        img_map[8:16, 8:16] = np.arange(64).reshape([8, 8])[::-1, :] + 29 * 64
        img_map[8:16, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 30 * 64
        img_map[8:16, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 31 * 64
        img_map[8:16, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 32 * 64
        img_map[8:16, 40:48] = np.arange(64).reshape([8, 8])[::-1, :] + 33 * 64

        img_map[16:24, 0:8] = np.arange(64).reshape([8, 8])[::-1, :] + 22 * 64
        img_map[16:24, 8:16] = np.arange(64).reshape([8, 8])[::-1, :] + 23 * 64
        img_map[16:24, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 24 * 64
        img_map[16:24, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 25 * 64
        img_map[16:24, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 26 * 64
        img_map[16:24, 40:48] = np.arange(64).reshape([8, 8])[::-1, :] + 27 * 64
        img_map[16:24, 48:56] = np.arange(64).reshape([8, 8])[::-1, :] + 28 * 64

        img_map[24:32, 0:8] = np.arange(64).reshape([8, 8])[::-1, :] + 15 * 64
        img_map[24:32, 8:16] = np.arange(64).reshape([8, 8])[::-1, :] + 16 * 64
        img_map[24:32, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 17 * 64
        img_map[24:32, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 18 * 64
        img_map[24:32, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 19 * 64
        img_map[24:32, 40:48] = np.arange(64).reshape([8, 8])[::-1, :] + 20 * 64
        img_map[24:32, 48:56] = np.arange(64).reshape([8, 8])[::-1, :] + 21 * 64

        img_map[32:40, 0:8] = np.arange(64).reshape([8, 8])[::-1, :] + 8 * 64
        img_map[32:40, 8:16] = np.arange(64).reshape([8, 8])[::-1, :] + 9 * 64
        img_map[32:40, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 10 * 64
        img_map[32:40, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 11 * 64
        img_map[32:40, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 12 * 64
        img_map[32:40, 40:48] = np.arange(64).reshape([8, 8])[::-1, :] + 13 * 64
        img_map[32:40, 48:56] = np.arange(64).reshape([8, 8])[::-1, :] + 14 * 64

        img_map[40:48, 8:16] = np.arange(64).reshape([8, 8])[::-1, :] + 3 * 64
        img_map[40:48, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 4 * 64
        img_map[40:48, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 5 * 64
        img_map[40:48, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 6 * 64
        img_map[40:48, 40:48] = np.arange(64).reshape([8, 8])[::-1, :] + 7 * 64

        img_map[48:56, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 0 * 64
        img_map[48:56, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 1 * 64
        img_map[48:56, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 2 * 64

        img_map = img_map + 1

        # This is set to int because no oversampling is done
        mapping_matrix3d = np.zeros((self.num_pixels['ASTRICam'] + 1,
                                     self.image_shapes['ASTRICam'][0],
                                     self.image_shapes['ASTRICam'][1]), dtype=int)

        # offset to the center:
        delta_x = int((self.image_shapes['ASTRICam'][1] - 56) / 2)
        delta_y = int((self.image_shapes['ASTRICam'][0] - 56) / 2)

        for x in range(56):
            for y in range(56):
                if img_map[x, y] > 0:
                    mapping_matrix3d[img_map[x, y], x + delta_x, y + delta_y] = 1

        sparse_map_mat = csr_matrix(mapping_matrix3d.reshape(self.num_pixels['ASTRICam'] + 1,
                                                             self.image_shapes['ASTRICam'][0]*
                                                             self.image_shapes['ASTRICam'][1]))

        return sparse_map_mat

    def generate_table_HESS(self, camera_type):
        """
            Function returning HESS-I or HESS-II mapping table (used to index into the trace when converting from trace to image).
        """
        if (camera_type=='HESS-I'):
            image_dim=72
            img_map = np.full([image_dim,image_dim], 0, dtype=int)
            blocks_num=8
            start_x=[55,63,71,71,71,71,63,55]
            start_y=[68,60,52,44,36,28,20,12]
            column_num_per_block = [20,28,36,36,36,36,28,20]

        if (camera_type=='HESS-II'):
            image_dim=104
            img_map = np.full([image_dim,image_dim], 0, dtype=int)
            blocks_num=12
            start_x=[71,87,95,103,103,103,103,103,103,95,87,71]
            start_y=[100,92,84,76,68,60,52,44,36,28,20,12]
            column_num_per_block = [20,36,44,52,52,52,52,52,52,44,36,20]

        pixel_weight = 1.0/4 #divide each pixel intensity into 4 sub pixels
        pixel_index = 1
        for block in np.arange(0,blocks_num):
            x = start_x[block]
            for i in np.arange(1,column_num_per_block[block]+1):

                if (i % 2 == 0):
                    y = start_y[block]-1
                else:
                    y = start_y[block]

                for j in np.arange(0,4):
                    #Assign the camera pixel index to the four image pixels
                    img_map[x,y]=pixel_index
                    img_map[x,y-1]=pixel_index
                    img_map[x-1,y]=pixel_index
                    img_map[x-1,y-1]=pixel_index
                    #Update x position and pixel_index
                    pixel_index+=1
                    y-=2
                #Update y position
                x-=2

        mapping_matrix3d = np.zeros((self.num_pixels[camera_type] + 1,
                                     self.image_shapes[camera_type][0],
                                     self.image_shapes[camera_type][1]), dtype=float)

        # offset to the center:
        delta_x = int((self.image_shapes[camera_type][1] - image_dim) / 2)
        delta_y = int((self.image_shapes[camera_type][0] - image_dim) / 2)

        for x in range(image_dim):
            for y in range(image_dim):
                if img_map[x, y] > 0:
                    mapping_matrix3d[img_map[x, y], x + delta_x, y + delta_y] = pixel_weight

        sparse_map_mat = csr_matrix(mapping_matrix3d.reshape(self.num_pixels[camera_type] + 1,
                                                     self.image_shapes[camera_type][0]*
                                                     self.image_shapes[camera_type][1]))

        return sparse_map_mat

    def generate_table_generic(self, camera_type, pixel_weight=1.0/4):
        if self.hex_conversion_algorithm == 'oversampling':
            # Note that this only works for Hex cams
            # Get telescope pixel positions for the given tel type
            pos = self.pixel_positions[camera_type]

            # Get relevant parameters
            output_dim = self.image_shapes[camera_type][0]
            num_pixels = self.num_pixels[camera_type]
            pixel_length = self.pixel_lengths[camera_type]

            # For LST:LSTCam and MST:NectarCam cameras, rotate by a fixed amount to
            # align for oversampling
            if camera_type in ["LSTCam", "NectarCam"]:
                pos = self.rotate_cam(pos)

            # Compute mapping matrix
            pos_int = pos / pixel_length * 2
            pos_int[0, :] = pos_int[0, :] / np.sqrt(3) * 2
            # below put the image in the corner
            pos_int[0, :] -= np.min(pos_int[0, :])
            pos_int[1, :] -= np.min(pos_int[1, :])
            p0_lim = np.max(pos_int[0, :]) - np.min(pos_int[0, :])
            p1_lim = np.max(pos_int[1, :]) - np.min(pos_int[1, :])
            if output_dim < p0_lim or output_dim < p1_lim:
                print("Danger! output image shape too small, will be cropped!")
            # below put the image in the center
            if camera_type in ["FACT"]:
                pos_int[0, :] += (output_dim - p0_lim) / 2. - 1.0
                pos_int[1, :] += (output_dim - p1_lim - 0.8) / 2. - 1.0
            else:
                pos_int[0, :] += (output_dim - p0_lim) / 2.
                pos_int[1, :] += (output_dim - p1_lim - 0.8) / 2.


            mapping_matrix = np.zeros((num_pixels + 1, output_dim, output_dim), dtype=float)

            for i in range(num_pixels):
                x, y = pos_int[:, i]
                x_S = int(round(x))
                x_L = x_S + 1
                y_S = int(round(y))
                y_L = y_S + 1
                # leave 0 for padding, mapping matrix from 1 to 499
                #mapping_matrix[i + 1, x_S:x_L + 1, y_S:y_L + 1] = pixel_weight
                mapping_matrix[i + 1, y_S:y_L + 1, x_S:x_L + 1] = pixel_weight

            # make sparse matrix of shape (num_pixels + 1, output_dim * output_dim)
            mapping_matrix = csr_matrix(mapping_matrix.reshape(num_pixels + 1, output_dim * output_dim))

        else:
            raise NotImplementedError("Cannot convert hexagonal camera image without valid conversion algorithm.")

        return mapping_matrix

    def rotate_cam(self, pos):
        rotation_matrix = np.matrix([[0.98198181, 0.18897548],
                             [-0.18897548, 0.98198181]], dtype=float)
        pos_rotated = np.squeeze(np.asarray(np.dot(rotation_matrix, pos)))

        return pos_rotated

    def rebinning(self):
        # placeholder
        raise NotImplementedError

    # internal methods to create pixel pos numpy files
    @staticmethod
    def __get_pos_from_h5(tel_table, camera_type="FlashCam", write=False, outfile=None):

        CAMERA_TYPE_TO_TEL_TYPE = {
            'LSTCam': 'LST:LSTCam',
            'FlashCam': 'MST:FlashCam',
            'NectarCam': 'MST:NectarCam',
            'SCTCam': 'SCT:SCTCam',
            'DigiCam': 'SST:DigiCam',
            'CHEC': 'SST:CHEC',
            'ASTRICam': 'SST:ASTRICam',
            'VERITAS': 'MST:VERITAS',
            'MAGICCam': 'MST:MAGICCam',
            'FACT': 'SST:FACT',
            'HESS-I': 'MST:HESS-I',
            'HESS-II': 'LST:HESS-II'
        }

        # TEMPORARY: Map camera type to (old) telescope type
        try:
            selected_tel_rows = np.array([row.nrow for row in tel_table.where('tel_type=={}'.format(CAMERA_TYPE_TO_TEL_TYPE[camera_type]))])[0]
        except:
            selected_tel_rows = np.array([row.nrow for row in tel_table.where('tel_type=={}'.format(NEW_TEL_NAMES_TO_OLD[CAMERA_TYPE_TO_TEL_TYPE[camera_type]]))])[0]

        pixel_pos = tel_table.cols.pixel_pos[selected_tel_rows]
        if write:
            if outfile is None:
                #outfile = "pixel_pos_files/{}_pos.npy".format(camera_type)
                outfile = os.path.join(os.path.dirname(__file__), "pixel_pos_files/{}_pos.npy".format(camera_type))
            np.save(outfile, pixel_pos)
        return pixel_pos

    @staticmethod
    def create_pix_pos_files(data_file):
        import tables # expect this to be run very rarely...

        with tables.open_file(data_file, "r") as f:
            tel_table = f.root.Telescope_Info
            for row in tel_table.iterrows():
                ImageMapper.__get_pos_from_h5(tel_table, camera_type=row[1].decode("utf-8"), write=True)

    def __read_pix_pos_files(self, camera_type):
        if camera_type in self.pixel_lengths:
            #infile = "pixel_pos_files/{}_pos.npy".format(camera_type)
            infile = os.path.join(os.path.dirname(__file__), "pixel_pos_files/{}_pos.npy".format(camera_type))
            return np.load(infile)
        else:
            logger.error("Camera type {} isn't supported.".format(camera_type))
            return False
