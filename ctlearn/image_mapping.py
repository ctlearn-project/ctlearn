import numpy as np
import logging
import threading
import os 
import cv2
    
from scipy import spatial
from scipy.sparse import csr_matrix
from collections import Counter

logger = logging.getLogger(__name__)

# Multithread-safe PyTables open and close file functions
# See http://www.pytables.org/latest/cookbook/threading.html
lock = threading.Lock()

class ImageMapper():

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
                 camera_type=None,
                 hex_conversion_algorithm=None,
                 interpolation_image_shape=None,
                 padding=None,
                 use_peak_times=False):
        """
        :param camera_type:  an array of strings specifying the camera types
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
            'LSTCam': (110, 110, 1),
            'FlashCam': (112, 112, 1),
            'NectarCam': (110, 110, 1),
            'SCTCam': (120, 120, 1),
            'DigiCam': (96, 96, 1),
            'CHEC': (48, 48, 1),
            'ASTRICam': (56, 56, 1),
            'VERITAS': (54, 54, 1),
            'MAGICCam': (78, 78, 1),
            'FACT': (90,90,1),
            'HESS-I': (72,72,1),
            'HESS-II': (104,104,1)
            }
        
        # Camera_type
        self.camera_type = camera_type
        if self.camera_type is None:
            self.camera_type = ['LSTCam', 'FlashCam', 'NectarCam', 'DigiCam', 'SCTCam', 'CHEC', 'ASTRICam', 'VERITAS', 'MAGICCam', 'FACT','HESS-I','HESS-II']
        
        # Hexagonal conversion algorithm
        self.hex_conversion_algorithm = hex_conversion_algorithm
        # Interpolation image shape
        self.interpolation_image_shape = interpolation_image_shape
        # Paddin
        self.padding = padding
        # Pixel positions and mapping tables initialization
        self.pixel_positions = {}
        self.pixel_rotation = {}
        self.mapping_tables = {}

        for camtype in self.camera_type:
            # Read pixel positions from fits file
            self.pixel_positions[camtype],self.pixel_rotation[camtype] = self.__read_pix_pos_from_fits(camtype)

            if hex_conversion_algorithm is None:
                self.hex_conversion_algorithm = {camtype:'oversampling'}
            elif camtype not in self.hex_conversion_algorithm.keys():
                self.hex_conversion_algorithm[camtype] = 'oversampling'
            
            if self.hex_conversion_algorithm[camtype] not in ['oversampling', 'rebinning', 'nearest_interpolation', 'bilinear_interpolation', 'bicubic_interpolation']:
                raise NotImplementedError("Hex conversion algorithm {} is not implemented.".format(hex_conversion_algorithm[camtype]))
            elif self.hex_conversion_algorithm[camtype] not in ['oversampling', 'rebinning', 'nearest_interpolation'] and camtype in ['SCTCam','CHEC','ASTRICam']:
                raise ValueError("Sorry! Camera type {} isn\'t supported with the conversion algorithm \'{}\'.".format(camtype,self.hex_conversion_algorithm[camtype]))
            
            if interpolation_image_shape is None:
                self.interpolation_image_shape = {camtype:self.image_shapes[camtype]}
            elif camtype not in interpolation_image_shape.keys():
                self.interpolation_image_shape[camtype] = self.image_shapes[camtype]

            if self.hex_conversion_algorithm[camtype] not in ['oversampling']:
                self.image_shapes[camtype] = self.interpolation_image_shape[camtype]

            if padding is None:
                self.padding = {camtype:0}
            elif camtype not in padding.keys():
                self.padding[camtype] = 0
            # A default padding is necessary for the hex_conversion_algorithm.
            self.default_pad = 2
            if self.hex_conversion_algorithm in ['bicubic_interpolation']:
                self.default_pad = 3

            if self.hex_conversion_algorithm[camtype] not in ['oversampling'] or camtype in ['ASTRICam', 'CHEC', 'SCTCam']:
                self.image_shapes[camtype] = (
                    self.image_shapes[camtype][0] + (self.padding[camtype] + self.default_pad) * 2,
                    self.image_shapes[camtype][1] + (self.padding[camtype] + self.default_pad) * 2,
                    self.image_shapes[camtype][2]
                )
            else:
                self.image_shapes[camtype] = (
                    self.image_shapes[camtype][0] + (self.padding[camtype] + self.default_pad*2) * 2,
                    self.image_shapes[camtype][1] + (self.padding[camtype] + self.default_pad*2) * 2,
                    self.image_shapes[camtype][2]
                )

            if use_peak_times:
                self.image_shapes[camtype] = (
                    self.image_shapes[camtype][0],
                    self.image_shapes[camtype][1],
                    2 # number of channels
                )

            self.mapping_tables[camtype] = self.generate_table(camtype)

            
    def map_image(self, pixels, camera_type):
        """
        :param pixels: a numpy array of values for each pixel, in order of pixel index.
                       For future reference: 
                       The array should have dimensions [N_pixels, N_channels] where N_channels is e.g., 
                       1 when just using charges and 2 when using charges and peak arrival times. 
        :param camera_type: a string specifying the telescope type as defined in the HDF5 format,
                        e.g., 'SCTCam' for SCT data, which is the only currently implemented telescope type.
        :return: a numpy array of shape [img_width, img_length, N_channels]
        """
        # Get relevant parameters
        output_dim = self.image_shapes[camera_type][0]
        default_pad = self.default_pad
        map_tab = self.mapping_tables[camera_type]
        hex_algo = self.hex_conversion_algorithm[camera_type]
        n_channels = pixels.shape[1]
        
        # We reshape each channel and then stack the result
        result = []
        for channel in range(n_channels):
            vector = pixels[:,channel]
            image_2D = (vector.T @ map_tab).reshape(self.image_shapes[camera_type][0],
                                                                    self.image_shapes[camera_type][1], 1)
            if default_pad != 0:
                if hex_algo not in ['oversampling'] or camera_type in ['ASTRICam', 'CHEC', 'SCTCam']:
                    image_2D = image_2D[default_pad:output_dim-default_pad, default_pad:output_dim-default_pad]
                    self.image_shapes[camera_type] = (
                        self.image_shapes[camera_type][0] - default_pad * 2,
                        self.image_shapes[camera_type][1] - default_pad * 2,
                        self.image_shapes[camera_type][2]
                    )
                else:
                    image_2D = image_2D[default_pad*2:output_dim-default_pad*2, default_pad*2:output_dim-default_pad*2]
                    self.image_shapes[camera_type] = (
                        self.image_shapes[camera_type][0] - default_pad * 4,
                        self.image_shapes[camera_type][1] - default_pad * 4,
                        self.image_shapes[camera_type][2]
                    )
            result.append(image_2D)
        telescope_image = np.concatenate(result, axis = -1)
        return telescope_image

    def generate_table(self, camera_type):
        # Get relevant parameters
        output_dim = self.image_shapes[camera_type][0]
        num_pixels = self.num_pixels[camera_type]
        # Get telescope pixel positions and padding for the given tel type
        pos = self.pixel_positions[camera_type]
        pad = self.padding[camera_type]
        hex_algo = self.hex_conversion_algorithm[camera_type]
        pos = self.slice_pixelPos(pos,num_pixels)
        
        # Creating the hexagonal and the output grid for the conversion methods.
        grid_size_factor = 1
        if hex_algo in ['rebinning']:
            grid_size_factor = 10
        hex_grid, table_grid = self.get_grids(pos, camera_type, grid_size_factor)
        
        # Oversampling and nearest interpolation
        if hex_algo in ['oversampling', 'nearest_interpolation']:
            # Finding the nearest point in the hexagonal grid for each point in the square grid
            tree = spatial.cKDTree(hex_grid)
            nn_index = np.reshape(tree.query(table_grid)[1],(output_dim, output_dim))
            if hex_algo in ['oversampling'] and camera_type not in ['ASTRICam', 'CHEC', 'SCTCam']:
                pixel_weight = 1/4
            else:
                pixel_weight = 1
            mapping_matrix3d = np.zeros((hex_grid.shape[0]+1,output_dim, output_dim))
            for y_grid in np.arange(0, output_dim, 1):
                for x_grid in np.arange(0, output_dim, 1):
                    mapping_matrix3d[nn_index[y_grid][x_grid]+1][y_grid][x_grid] = pixel_weight
        
            mapping_matrix3d = mapping_matrix3d[0:num_pixels+1]
            
            # Rotating the camera back to the original orientation
            if camera_type in ['LSTCam', 'NectarCam', 'MAGICCam']:
                for i in np.arange(0,mapping_matrix3d.shape[0],1):
                    mapping_matrix3d[i] = self.rotate_image(mapping_matrix3d[i],camera_type,self.pixel_rotation[camera_type])

            sparse_map_mat = csr_matrix(mapping_matrix3d.reshape(mapping_matrix3d.shape[0],
                                                                 self.image_shapes[camera_type][0]*
                                                                 self.image_shapes[camera_type][1]))

            return sparse_map_mat
        
        # Rebinning (approximation)
        elif hex_algo in ['rebinning']:
            # Finding the nearest point in the hexagonal grid for each point in the square grid
            tree = spatial.cKDTree(hex_grid)
            nn_index = np.reshape(tree.query(table_grid)[1],(output_dim*grid_size_factor, output_dim*grid_size_factor))
            
            # Calculating the overlapping area/weights for each square pixel
            mapping_matrix3d = np.zeros((hex_grid.shape[0]+1,output_dim, output_dim))
            for y_grid in np.arange(0, output_dim*grid_size_factor, grid_size_factor):
                for x_grid in np.arange(0, output_dim*grid_size_factor, grid_size_factor):
                    counter = Counter(np.reshape(nn_index[y_grid:y_grid+grid_size_factor, x_grid:x_grid+grid_size_factor],-1))
                    pixel_index = np.array(list(counter.keys()))+1
                    weights = list(counter.values())/np.sum(list(counter.values()))
                    for key in np.arange(0,len(pixel_index),1):
                        mapping_matrix3d[pixel_index[key]][int(y_grid/grid_size_factor)][int(x_grid/grid_size_factor)] = weights[key]

            mapping_matrix3d = mapping_matrix3d[0:num_pixels+1]
            
            # Normalization (approximation) of the mapping table
            norm_factor = 0
            for i in np.arange(1,mapping_matrix3d.shape[0],1):
                norm_factor += np.sum(mapping_matrix3d[i])
            norm_factor /= float(num_pixels)
            for i in np.arange(1,mapping_matrix3d.shape[0],1):
                mapping_matrix3d[i] /= norm_factor
            
            # Rotating the camera back to the original orientation
            if camera_type in ['LSTCam', 'NectarCam', 'MAGICCam']:
                for i in np.arange(0,mapping_matrix3d.shape[0],1):
                    mapping_matrix3d[i] = self.rotate_image(mapping_matrix3d[i],camera_type,self.pixel_rotation[camera_type])
            
            sparse_map_mat = csr_matrix(mapping_matrix3d.reshape(mapping_matrix3d.shape[0],
                                                                 self.image_shapes[camera_type][0]*
                                                                 self.image_shapes[camera_type][1]))
            return sparse_map_mat
                
        # Bilinear interpolation
        elif hex_algo in ['bilinear_interpolation']:
            # Finding the nearest point in the hexagonal grid for each point in the square grid
            tree = spatial.cKDTree(hex_grid)
            nn_index = np.reshape(tree.query(table_grid)[1],(output_dim, output_dim))
            tri = spatial.Delaunay(hex_grid)
                
            table_simplex = tri.simplices[tri.find_simplex(table_grid)]
            table_simplex_points = hex_grid[table_simplex]
            weights = self.get_weights(table_simplex_points, table_grid)
            weights = np.reshape(weights, (output_dim, output_dim, weights.shape[1]))
            table_simplex = np.reshape(table_simplex,(output_dim, output_dim, table_simplex.shape[1]))
            
            mapping_matrix3d = np.zeros((hex_grid.shape[0]+1, output_dim, output_dim))
            for i in np.arange(0,output_dim,1):
                for j in np.arange(0,output_dim,1):
                    for k in np.arange(0,3,1):
                        mapping_matrix3d[table_simplex[j][i][k]+1][j][i] = weights[j][i][k]

            mapping_matrix3d = mapping_matrix3d[0:num_pixels+1]
            
            # Normalization (approximation) of the mapping table
            norm_factor = 0
            for i in np.arange(1,mapping_matrix3d.shape[0],1):
                norm_factor += np.sum(mapping_matrix3d[i])
            norm_factor /= float(num_pixels)
            for i in np.arange(1,mapping_matrix3d.shape[0],1):
                mapping_matrix3d[i] /= norm_factor
            
            # Mask interpolation
            for j in np.arange(0,nn_index.shape[0],1):
                for k in np.arange(0,nn_index.shape[1],1):
                    if nn_index[k][j] >= num_pixels:
                        for i in np.arange(1,mapping_matrix3d.shape[0],1):
                            mapping_matrix3d[i][k][j] = 0.0

            # Rotating the camera back to the original orientation
            if camera_type in ['LSTCam', 'NectarCam', 'MAGICCam']:
                for i in np.arange(0,mapping_matrix3d.shape[0],1):
                    mapping_matrix3d[i] = self.rotate_image(mapping_matrix3d[i],camera_type,self.pixel_rotation[camera_type])

            sparse_map_mat = csr_matrix(mapping_matrix3d.reshape(mapping_matrix3d.shape[0],
                                                                self.image_shapes[camera_type][0]*
                                                                self.image_shapes[camera_type][1]))
                                                                                                            
            return sparse_map_mat
                                                                                                                
        # Bicubic interpolation
        elif hex_algo in ['bicubic_interpolation']:
            #
            #                 /\        /\
            #                /  \      /  \
            #               /    \    /    \
            #              / 2NN  \  / 2NN  \
            #             /________\/________\
            #            /\        /\        /\
            #           /  \  NN  /  \  NN  /  \
            #          /    \    /    \    /    \
            #         / 2NN  \  /  .   \  /  2NN \
            #        /________\/________\/________\
            #                 /\        /\
            #                /  \  NN  /  \
            #               /    \    /    \
            #              / 2NN  \  / 2NN  \
            #             /________\/________\
            #
            
            # Finding the nearest point in the hexagonal grid for each point in the square grid
            tree = spatial.cKDTree(hex_grid)
            nn_index = np.reshape(tree.query(table_grid)[1],(output_dim, output_dim))
            tri = spatial.Delaunay(hex_grid)
                                                                                                                    
            # Get all relevant simplex indices
            simplex_index = tri.find_simplex(table_grid)
            simplex_index_NN = tri.neighbors[simplex_index]
            simplex_index_2NN = tri.neighbors[simplex_index_NN]
            
            table_simplex = tri.simplices[simplex_index]
            table_simplex_points = hex_grid[table_simplex]

            # NN
            weights_NN = []
            simplexes_NN = []
            for i in np.arange(0,simplex_index.shape[0],1):
                if -1 in simplex_index_NN[i] or all(ind >= num_pixels for ind in table_simplex[i]):
                    w = np.array([0,0,0])
                    weights_NN.append(w)
                    corner_simplexes_2NN = np.array([-1,-1,-1])
                    simplexes_NN.append(corner_simplexes_2NN)
                else:
                    corner_points_NN, corner_simplexes_NN = self.get_triangle(tri, hex_grid, simplex_index_NN[i], table_simplex[i])
                    target = table_grid[i]
                    target = np.expand_dims(target, axis=0)
                    w = self.get_weights(corner_points_NN, target)
                    w = np.squeeze(w, axis=0)
                    weights_NN.append(w)
                    simplexes_NN.append(corner_simplexes_NN)

            weights_NN = np.array(weights_NN)
            simplexes_NN = np.array(simplexes_NN)

            # 2NN
            weights_2NN = []
            simplexes_2NN = []
            for i in np.arange(0,3,1):
                weights = []
                simplexes = []
                for j in np.arange(0,simplex_index.shape[0],1):
                    table_simplex_NN = tri.simplices[simplex_index_NN[j][i]]
                    if -1 in simplex_index_2NN[j][i] or -1 in simplex_index_NN[j] or all(ind >= num_pixels for ind in table_simplex_NN):
                        w = np.array([0,0,0])
                        weights.append(w)
                        corner_simplexes_2NN = np.array([-1,-1,-1])
                        simplexes.append(corner_simplexes_2NN)
                    else:
                        corner_points_2NN, corner_simplexes_2NN = self.get_triangle(tri, hex_grid, simplex_index_2NN[j][i], table_simplex_NN)
                        target = table_grid[j]
                        target = np.expand_dims(target, axis=0)
                        w = self.get_weights(corner_points_2NN, target)
                        w = np.squeeze(w, axis=0)
                        weights.append(w)
                        simplexes.append(corner_simplexes_2NN)
            
                weights = np.array(weights)
                simplexes = np.array(simplexes)
                weights_2NN.append(weights)
                simplexes_2NN.append(simplexes)

            weights_2NN.append(weights_NN)
            simplexes_2NN.append(simplexes_NN)
            weights_2NN = np.array(weights_2NN)
            simplexes_2NN = np.array(simplexes_2NN)
            weights_2NN = np.reshape(weights_2NN,(weights_2NN.shape[0], output_dim, output_dim, weights_2NN.shape[2]))
            simplexes_2NN = np.reshape(simplexes_2NN,(simplexes_2NN.shape[0], output_dim, output_dim, simplexes_2NN.shape[2]))

            mapping_matrix3d = np.zeros((hex_grid.shape[0]+1, output_dim, output_dim))
            for i in np.arange(0,4,1):
                for j in np.arange(0,output_dim,1):
                    for k in np.arange(0,output_dim,1):
                        for l in np.arange(0,3,1):
                            mapping_matrix3d[simplexes_2NN[i][k][j][l]+1][k][j] = weights_2NN[i][k][j][l]/4

            mapping_matrix3d = mapping_matrix3d[0:num_pixels+1]

            # Normalization (approximation) of the mapping table
            norm_factor = 0
            for i in np.arange(1,mapping_matrix3d.shape[0],1):
                norm_factor += np.sum(mapping_matrix3d[i])
            norm_factor /= float(num_pixels)
            for i in np.arange(1,mapping_matrix3d.shape[0],1):
                mapping_matrix3d[i] /= norm_factor
            
            # Mask interpolation
            for j in np.arange(0,nn_index.shape[0],1):
                for k in np.arange(0,nn_index.shape[1],1):
                    if nn_index[k][j] >= num_pixels:
                        for i in np.arange(1,mapping_matrix3d.shape[0],1):
                            mapping_matrix3d[i][k][j] = 0.0

            # Rotating the camera back to the original orientation
            if camera_type in ['LSTCam', 'NectarCam', 'MAGICCam']:
                for i in np.arange(0,mapping_matrix3d.shape[0],1):
                    mapping_matrix3d[i] = self.rotate_image(mapping_matrix3d[i],camera_type,self.pixel_rotation[camera_type])

            sparse_map_mat = csr_matrix(mapping_matrix3d.reshape(mapping_matrix3d.shape[0],
                                                                self.image_shapes[camera_type][0]*
                                                                self.image_shapes[camera_type][1]))
                                                                                                                                                                                                                                
            return sparse_map_mat

    def get_triangle(self, tri, hex_grid, simplex_index_NN, table_simplex):
        """
        :param tri: a Delaunay triangulation.
        :param hex_grid: a 2D numpy array (hexagonal grid).
        :param simplex_index_NN: a numpy array containing the indexes of the three neighboring simplexes.
        :param table_simplex: a numpy array containing the three indexes (hexaganol grid) of the target simplex.
        :return: two numpy array containing the three corner points and simplexes.
        """
        # This function is calculating the corner points (marked as 'X') and simplexes
        # for the nearest neighbor (NN) triangles. The function returns a bigger triangle,
        # which contains four Delaunay triangles.
        #
        #            X--------------------X
        #             \        /\        /
        #              \  NN  /  \  NN  /
        #               \    /    \    /
        #                \  /  .   \  /
        #                 \/________\/
        #                  \        /
        #                   \  NN  /
        #                    \    /
        #                     \  /
        #                      X
        #
        
        corner_points = []
        corner_simplexes = []
        for neighbors in np.arange(0,3,1):
            table_simplex_NN = tri.simplices[simplex_index_NN[neighbors]]
            simplex = np.array(list(set(table_simplex_NN) - set(table_simplex)))
            simplex = np.squeeze(simplex, axis=0)
            corner_simplexes.append(simplex)
            corner_points.append(hex_grid[simplex])
        corner_points = np.array(corner_points)
        corner_simplexes = np.array(corner_simplexes)
        corner_points = np.expand_dims(corner_points, axis=0)
        return corner_points, corner_simplexes

    def get_weights(self, p, target):
        """
        :param p: a numpy array of shape (i,3,2) for three 2D points (one triangual). The index i means that one can calculate the weights for multiply trianguals with one function call.
        :param target: a numpy array of shape (i,2) for one target 2D point.
        :return: a numpy array of shape (i,3) containing the three weights.
        """
       
        #       Barycentric coordinates:
        #                 (x3,y3)
        #                   .
        #                  / \
        #                 /   \
        #                /     \
        #               /       \
        #              /         \
        #             /        .  \
        #            /       (x,y) \
        #    (x1,y1)._______________.(x2,y2)
        #
        #       x = w1*x1 + w2*x2 + w3*x3
        #       y = w1*y1 + w2*y2 + w3*y3
        #       1 = w1 + w2 + w3
        #
        #       -> Rearranging:
        #              (y2-y3)*(x-x3)+(x3-x2)*(y-y3)
        #       w1 = ---------------------------------
        #             (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
        #
        #              (y3-y1)*(x-x3)+(x1-x3)*(y-y3)
        #       w2 = ---------------------------------
        #             (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
        #
        #       w3 = 1 - w1 - w2
        #
        
        weights = []
        for i in np.arange(0,p.shape[0],1):
            w=[0,0,0]
            divisor = float(((p[i][1][1]-p[i][2][1])*(p[i][0][0]-p[i][2][0])+(p[i][2][0]-p[i][1][0])*(p[i][0][1]-p[i][2][1])))
            w[0] = float(((p[i][1][1]-p[i][2][1])*(target[i][0]-p[i][2][0])+(p[i][2][0]-p[i][1][0])*(target[i][1]-p[i][2][1])))/divisor
            w[1] = float(((p[i][2][1]-p[i][0][1])*(target[i][0]-p[i][2][0])+(p[i][0][0]-p[i][2][0])*(target[i][1]-p[i][2][1])))/divisor
            w[2] = 1-w[0]-w[1]
            weights.append(w)
        return np.array(weights)

    def get_grids(self, pos, camera_type, grid_size_factor):
        """
        :param pos: a 2D numpy array of pixel positions, which were taken from the CTApipe.
        :param camera_type: a string specifying the camera type
        :param grid_size_factor: a number specifying the grid size of the output grid. Only if 'rebinning' is selected, this factor differs from 1.
        :return: two 2D numpy arrays (hexagonal grid and squared output grid)
        """
    
        # Get relevant parameters
        output_dim = self.image_shapes[camera_type][0]
        num_pixels = self.num_pixels[camera_type]
        pad = self.padding[camera_type]
        default_pad = self.default_pad
        hex_algo = self.hex_conversion_algorithm[camera_type]
        
        x=np.around(pos[0],decimals=3)
        y=np.around(pos[1],decimals=3)
        
        x_ticks=np.unique(x).tolist()
        y_ticks=np.unique(y).tolist()
        
        if camera_type in ['CHEC', 'ASTRICam', 'SCTCam']:
            
            if camera_type in ['CHEC']:
                # The algorithm doesn't work with the CHEC camera. Additional smoothing
                # for the 'x_ticks' and 'y_ticks' array for CHEC pixel positions.
                num_x_ticks = len(x_ticks)
                remove_x_val = []
                change_x_val = []
                for i in np.arange(0,num_x_ticks-1,1):
                    if np.abs(x_ticks[i]-x_ticks[i+1]) <= 0.002:
                        remove_x_val.append(x_ticks[i])
                        change_x_val.append(x_ticks[i+1])
                for j in np.arange(0,len(remove_x_val),1):
                    x_ticks.remove(remove_x_val[j])
                    for k in np.arange(0,len(x),1):
                        if x[k] == remove_x_val[j]:
                            x[k]=change_x_val[j]
            
                num_y_ticks = len(y_ticks)
                remove_y_val = []
                change_y_val = []
                for i in np.arange(0,num_y_ticks-1,1):
                    if np.abs(y_ticks[i]-y_ticks[i+1]) <= 0.002:
                        remove_y_val.append(y_ticks[i])
                        change_y_val.append(y_ticks[i+1])


                for j in np.arange(0,len(remove_y_val),1):
                    y_ticks.remove(remove_y_val[j])
                    for k in np.arange(0,len(y),1):
                        if y[k] == remove_y_val[j]:
                            y[k]=change_y_val[j]
        
            x_dist = np.around(abs(x_ticks[0]-x_ticks[1]),decimals=3)
            y_dist = np.around(abs(y_ticks[0]-y_ticks[1]),decimals=3)
            for i in np.arange(0,default_pad,1):
                x_ticks.append(np.around(x_ticks[-1]+x_dist,decimals=3))
                x_ticks.insert(0,np.around(x_ticks[0]-x_dist,decimals=3))
                y_ticks.append(np.around(y_ticks[-1]+y_dist,decimals=3))
                y_ticks.insert(0,np.around(y_ticks[0]-y_dist,decimals=3))

            virtual_pixel_x = []
            virtual_pixel_y = []
            for i in x_ticks:
                for j in y_ticks:
                    camPix=0
                    for m in np.arange(0,num_pixels,1):
                        if (x[m]==i and y[m]==j):
                            camPix=1
                    if (camPix==0):
                        virtual_pixel_x.append(i)
                        virtual_pixel_y.append(j)
    
            x = np.concatenate((x,np.array(virtual_pixel_x)))
            y = np.concatenate((y,np.array(virtual_pixel_y)))
            hex_grid = np.column_stack([x,y])
            
            if hex_algo in ['oversampling']:
                for i in np.arange(0,pad,1):
                    x_ticks.append(np.around(x_ticks[-1]+x_dist,decimals=3))
                    x_ticks.insert(0,np.around(x_ticks[0]-x_dist,decimals=3))
                    y_ticks.append(np.around(y_ticks[-1]+y_dist,decimals=3))
                    y_ticks.insert(0,np.around(y_ticks[0]-y_dist,decimals=3))
                x_grid, y_grid = np.meshgrid(x_ticks, y_ticks)
            else:
                x_pad = pad*(np.max(x_ticks)-np.min(x_ticks))/(output_dim-pad*2)
                y_pad = pad*(np.max(y_ticks)-np.min(y_ticks))/(output_dim-pad*2)
                xx = np.linspace(np.min(x_ticks)-x_pad, np.max(x_ticks)+x_pad, num=output_dim*grid_size_factor, endpoint=True)
                yy = np.linspace(np.min(y_ticks)-y_pad, np.max(y_ticks)+y_pad, num=output_dim*grid_size_factor, endpoint=True)
                x_grid, y_grid = np.meshgrid(xx, yy)

            x_grid = np.reshape(x_grid,-1)
            y_grid = np.reshape(y_grid,-1)
            output_grid = np.column_stack([x_grid, y_grid])
        else:
            if len(x_ticks) < len(y_ticks):
                first_ticks=x_ticks
                first_pos=x
                second_ticks=y_ticks
                second_pos=y
            else:
                first_ticks=y_ticks
                first_pos=y
                second_ticks=x_ticks
                second_pos=x
        
            dist_first = np.around(abs(first_ticks[0]-first_ticks[1]),decimals=3)
            dist_second = np.around(abs(second_ticks[0]-second_ticks[1]),decimals=3)
        
            if hex_algo in ['oversampling']:
                tick_diff = (len(first_ticks)*2 - len(second_ticks))
                tick_diff_each_side = np.array(int(tick_diff/2))
            else:
                tick_diff = 0
                tick_diff_each_side = 0
            for i in np.arange(0,tick_diff_each_side+default_pad*2,1):
                second_ticks.append(np.around(second_ticks[-1]+dist_second,decimals=3))
                second_ticks.insert(0,np.around(second_ticks[0]-dist_second,decimals=3))
            for i in np.arange(0,default_pad,1):
                first_ticks.append(np.around(first_ticks[-1]+dist_first,decimals=3))
                first_ticks.insert(0,np.around(first_ticks[0]-dist_first,decimals=3))

            if tick_diff % 2 != 0:
                second_ticks.insert(0,np.around(second_ticks[0]-dist_second,decimals=3))
            
            # Creating the virtual pixels outside of the camera.
            virtual_pixel_x,virtual_pixel_y = self.add_virtualPixels(first_ticks, second_ticks, first_pos, second_pos, num_pixels)

            first_pos=np.concatenate((first_pos,np.array(virtual_pixel_x)))
            second_pos=np.concatenate((second_pos,np.array(virtual_pixel_y)))

            if hex_algo in ['oversampling']:
                grid_first = []
                for i in first_ticks:
                    grid_first.append(i-dist_first/4.0)
                    grid_first.append(i+dist_first/4.0)
                grid_second = []
                for j in second_ticks:
                    grid_second.append(j+dist_second/2.0)
                # Padding
                dist_grid_first = np.around(abs(grid_first[0]-grid_first[1]),decimals=3)
                dist_grid_second = np.around(abs(grid_second[0]-grid_second[1]),decimals=3)
                for i in np.arange(0,pad,1):
                    grid_first.append(np.around(grid_first[-1]+dist_grid_first,decimals=3))
                    grid_first.insert(0,np.around(grid_first[0]-dist_grid_first,decimals=3))
                    grid_second.append(np.around(grid_second[-1]+dist_grid_second,decimals=3))
                    grid_second.insert(0,np.around(grid_second[0]-dist_grid_second,decimals=3))
            else:
                # Add corner
                first_pad = pad*(np.max(first_pos)-np.min(first_pos))/(output_dim-pad*2)
                second_pad = pad*(np.max(second_pos)-np.min(second_pos))/(output_dim-pad*2)
        
                minimum = min([np.min(first_pos) - first_pad,np.min(second_pos) - second_pad])
                maximum = max([np.max(first_pos) + first_pad,np.max(second_pos) + second_pad])
    
                first_pos=np.concatenate((first_pos,[minimum]))
                second_pos=np.concatenate((second_pos,[minimum]))
                first_pos=np.concatenate((first_pos,[minimum]))
                second_pos=np.concatenate((second_pos,[maximum]))
                first_pos=np.concatenate((first_pos,[maximum]))
                second_pos=np.concatenate((second_pos,[minimum]))
                first_pos=np.concatenate((first_pos,[maximum]))
                second_pos=np.concatenate((second_pos,[maximum]))
            
                grid_first = grid_second = np.linspace(minimum, maximum, num=output_dim*grid_size_factor, endpoint=True)
    
            if len(x_ticks) < len(y_ticks):
                hex_grid = np.column_stack([first_pos,second_pos])
                x_grid, y_grid = np.meshgrid(grid_first, grid_second)
            else:
                hex_grid = np.column_stack([second_pos,first_pos])
                x_grid, y_grid = np.meshgrid(grid_second, grid_first)
            x_grid = np.reshape(x_grid,-1)
            y_grid = np.reshape(y_grid,-1)
            output_grid = np.column_stack([x_grid, y_grid])
        
        return hex_grid,output_grid
    
    def slice_pixelPos(self, pos, num_pixels):
        slice_pos = []
        slice_pos.append(pos[0][0:num_pixels])
        slice_pos.append(pos[1][0:num_pixels])
        slice_pos = np.array(slice_pos)
        return slice_pos
    
    def add_virtualPixels(self, first_ticks, second_ticks, first_pos, second_pos, num_pixels):
        virtual_pixel_x = []
        virtual_pixel_y = []
        pixCounter=0
        for i in first_ticks[0::2]:
            for j in second_ticks[0::2]:
                camPix=0
                for m in np.arange(0,num_pixels,1):
                    if (first_pos[m]==i and second_pos[m]==j):
                        camPix=1
                        pixCounter+=1
                if (camPix==0):
                    virtual_pixel_x.append(i)
                    virtual_pixel_y.append(j)
        for i in first_ticks[1::2]:
            for j in second_ticks[1::2]:
                camPixel=0
                for m in np.arange(0,num_pixels,1):
                    if (first_pos[m]==i and second_pos[m]==j):
                        camPixel=1
                        pixCounter+=1
                if (camPixel==0):
                    virtual_pixel_x.append(i)
                    virtual_pixel_y.append(j)
                    
        if pixCounter==0:
            virtual_pixel_x = []
            virtual_pixel_y = []
            for i in first_ticks[1::2]:
                for j in second_ticks[0::2]:
                    camPix=0
                    for m in np.arange(0,num_pixels,1):
                        if (first_pos[m]==i and second_pos[m]==j):
                            camPix=1
                            pixCounter+=1
                    if (camPix==0):
                        virtual_pixel_x.append(i)
                        virtual_pixel_y.append(j)
            for i in first_ticks[0::2]:
                for j in second_ticks[1::2]:
                    camPixel=0
                    for m in np.arange(0,num_pixels,1):
                        if (first_pos[m]==i and second_pos[m]==j):
                            camPixel=1
                            pixCounter+=1
                    if (camPixel==0):
                        virtual_pixel_x.append(i)
                        virtual_pixel_y.append(j)

        return virtual_pixel_x,virtual_pixel_y

    def rotate_pixel_pos(self, pos, angle_deg):
        angle = angle_deg * np.pi/180.0
        rotation_matrix = np.matrix([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]], dtype=float)
        pos_rotated = np.squeeze(np.asarray(np.dot(rotation_matrix, pos)))
        return pos_rotated

    def rotate_image(self, image, camera_type, angle):
        image = np.expand_dims(image, axis=2)
        h = w = self.image_shapes[camera_type][0]
        center = (w/2.0, h/2.0)
        scale=1.0
        M = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(image, M, (w, h))
        return image
    
    # internal methods to create pixel pos numpy files 
    def __get_pos_from_h5(self, tel_table, camera_type="FlashCam", write=False, outfile=None):
        selected_tel_rows = np.array([row.nrow for row in tel_table.where('camera_type=={}'.format(camera_type))])[0]
        pixel_pos = tel_table.cols.pixel_pos[selected_tel_rows]
        if write:
            if outfile is None:
                #outfile = "pixel_pos_files/{}_pos.npy".format(camera_type)
                outfile = os.path.join(os.path.dirname(__file__), "pixel_pos_files/{}_pos.npy".format(camera_type))
            np.save(outfile, pixel_pos)
        return pixel_pos
    
    def __create_pix_pos_files(self, data_file):
        import tables # expect this to be run very rarely...

        with tables.open_file(data_file, "r") as f:
            tel_table = f.root.Telescope_Info
            for row in tel_table.iterrows():
                self.__get_pos_from_h5(tel_table, camera_type=row[1].decode("utf-8"), write=True)

    def __read_pix_pos_files(self, camtype):
        if camtype in self.num_pixels.keys():
            #infile = "pixel_pos_files/{}_pos.npy".format(camera_type)
            infile = os.path.join(os.path.dirname(__file__), "pixel_pos_files/{}_pos.npy".format(camtype))
            return np.load(infile)
        else:
            logger.error("Camera type {} isn't supported.".format(camtype))
            return False

    def __read_pix_pos_from_fits(self, camtype):
        from astropy.io import fits
        if camtype in self.num_pixels.keys():
            # Camera geometry fits files from cta-observatory/ctapipe-extra v0.2.16
            hdul = fits.open('pixel_pos_files/{}.camgeom.fits'.format(camtype))
            data = hdul[1].data
            header = hdul[1].header
            x = data.field('pix_x')
            y = data.field('pix_y')
            pos = np.column_stack([x,y]).T
            pixel_rot = 0.0
            # For LSTCam/NectarCam and MAGICCam, rotate by a fixed amount to
            # align with x and y axis.
            if camtype in ['LSTCam','NectarCam','MAGICCam']:
                pixel_rot = 90.0-header['PIX_ROT']
                pos = self.rotate_pixel_pos(pos, pixel_rot)
            return pos, pixel_rot
        else:
            logger.error("Camera type {} isn't supported.".format(camera_type))
            return False
