import numpy as np
import logging
import threading

from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# Multithread-safe PyTables open and close file functions
# See http://www.pytables.org/latest/cookbook/threading.html
lock = threading.Lock()


class ImageMapper():
    def __init__(self, image_mapping_settings):
        """   
        :param image_mapping_settings: (Hex converter algorithm, output image shape image_shapes, ...)
        """

        self.image_shapes = {
            'MSTS': (120, 120, 1),
            'VTS': (54, 54, 1),
            'MSTF': (120, 120, 1),
            'MSTN': (120, 120, 1),
            'LST': (120, 120, 1),
            'SST1': (100, 100, 1)
        }

        self.pixel_lengths = {'LST': 0.05, 'MSTF': 0.05, 'MSTN': 0.05, 'SST1': 0.0236}
        
        self.pixel_positions = {self.__read_pix_pos_files(tel_type) 
                for tel_type in self.pixel_lengths}

        self.num_pixels = {}
        for tel_type in self.pixel_lengths:
            pos_sq_ = pos[0, :] ** 2 + pos[1, :] ** 2
            self.num_pixels[tel_type] = pos_sq_[np.where(pos_sq_>0)].shape[0]

        self.mapping_tables = {
            'MSTS': self.generate_table_MSTS(),
            'VTS': self.generate_table_VTS(),
            'MSTF': self.generate_table_generic('MSTF'),
            'MSTN': self.generate_table_generic('MSTN'),
            'LST': self.generate_table_generic('LST'),
            'SST1': self.generate_table_generic('SST1')
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
        if telescope_type in self.image_shapes.keys():
            self.telescope_type = telescope_type
        else:
            raise ValueError('Sorry! Telescope type {} isn\'t supported.'.format(telescope_type))

        if telescope_type == "MSTS":
            # Create image by indexing into the trace using the mapping table, then adding a
            # dimension to given shape (length,width,1)
            telescope_image = np.expand_dims(pixels[self.mapping_tables[telescope_type]], 2)
        elif telescope_type == "VTS":
            telescope_image = (pixels.T @ self.mapping_tables[telescope_type]).reshape(self.image_shapes['VTS'][0],
                                                                                       self.image_shapes['VTS'][1], 1)
        elif telescope_type in ['LST', 'MSTF', 'MSTN', 'SST1']:
            telescope_image = (pixels.T @ self.mapping_tables[telescope_type]).reshape(self.image_shapes[telescope_type][0],
                                                                                       self.image_shapes[telescope_type][1], 1)
        
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
        # telescope hardcoded values
        num_pixels = 499
        pixel_side_len = 1.0 * np.sqrt(2)
        num_spirals = 13

        pixel_weight = 1.0/4 #divide each pixel intensity into 4 sub pixels
        
        pos = np.zeros((num_pixels, 2), dtype=float)
        delta_x = pixel_side_len * np.sqrt(2) / 2.
        delta_y = pixel_side_len * np.sqrt(2)
        
        pixel_index = 1

        # return mapping_matrix (54, 54)
        # leave 0 for padding, mapping matrix from 1 to 499
        mapping_matrix = np.zeros((num_pixels + 1, self.image_shapes['VTS'][0], self.image_shapes['VTS'][1]), dtype=float)

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
        for i in range(num_pixels):
            x, y = pos_shifted[i, :]
            x_L = int(round(x + pixel_side_len / 2.0))
            x_S = int(round(x - pixel_side_len / 2.0))
            y_L = int(round(y + pixel_side_len / 2.0))
            y_S = int(round(y - pixel_side_len / 2.0))
            
            # leave 0 for padding, mapping matrix from 1 to 499
            mapping_matrix[i + 1, x_S:x_L + 1, y_S:y_L + 1] = pixel_weight

        mapping_matrix = csr_matrix(mapping_matrix.reshape(num_pixels + 1, self.image_shapes['VTS'][0] * self.image_shapes['VTS'][1]))        
        
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
        MODULE_START_POSITIONS = [(((self.image_shapes['MSTS'][0] - MODULES_PER_ROW[j] *
                                     MODULE_DIM) / 2) +
                                   (MODULE_DIM * i), j * MODULE_DIM)
                                  for j in range(ROWS)
                                  for i in range(MODULES_PER_ROW[j])]

        table = np.zeros(shape=(self.image_shapes['MSTS'][0], self.image_shapes['MSTS'][1]), dtype=int)
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

    def generate_table_generic(self, tel_type, pixel_weight=1.0/4):
        # Get telescope pixel positions for the given tel type
        pos = self.pixel_positions[tel_type]

        # Get relevant parameters
        output_dim = self.image_shapes[tel_type][0]
        num_pixels = self.num_pixels[tel_type]
        pixel_length = self.pixel_lengths[tel_type]

        # For LST and MSTN cameras, rotate by a fixed amount to
        # align for oversampling
        if tel_type in ["LST", "MSTN"]:
            pos = self.rotate_cam(pos)

        # Compute mapping matrix
        pos_int = pos / pixel_length * 2 + 1
        pos_int[0, :] = pos_int[0, :] / np.sqrt(3) * 2
        pos_int[0, :] -= np.min(pos_int[0, :])
        pos_int[1, :] -= np.min(pos_int[1, :])

        mapping_matrix = np.zeros((num_pixels + 1, output_dim, output_dim), dtype=float)

        for i in range(num_pixels):
            x, y = pos_int[:, i]
            x_S = int(round(x))
            x_L = x_S + 1
            y_S = int(round(y))
            y_L = y_S + 1
            # leave 0 for padding, mapping matrix from 1 to 499
            mapping_matrix[i + 1, x_S:x_L + 1, y_S:y_L + 1] = pixel_weight

        # make sparse matrix of shape (num_pixels + 1, output_dim * output_dim)
        mapping_matrix = csr_matrix(mapping_matrix.reshape(num_pixels + 1, output_dim * output_dim))

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
    def __get_pos_from_h5(tel_table, tel_type="MSTF", write=False, outfile=None):
        selected_tel_rows = np.array([row.nrow for row in tel_table.where('tel_type=={}'.format(tel_type))])[0]
        pixel_pos = tel_table.cols.pixel_pos[selected_tel_rows]
        if write:
            if outfile is None:
                outfile = "{}_pos.npy".format(tel_type)
            np.save(outfile, pixel_pos)
        return pixel_pos
    
    def __create_pix_pos_files(data_file):
        import tables # expect this to be run very rarely...

        with tables.open_file(data_file, "r") as f:
            tel_table = f.root.Telescope_Info
            for row in tel_table.iterrows():
                self.__get_pos_from_h5(tel_table, tel=row[1].decode("utf-8"), write=True)

    def __read_pix_pos_files(tel_type):
        if tel_type in self.pixel_lengths: 
            infile = "{}_pos.npy".format(tel)
            return np.load(infile)
        else:
            logger.error("Telescope type {} isn't supported.".format(tel_type))
            return False
 

