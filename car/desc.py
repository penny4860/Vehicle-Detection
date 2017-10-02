
from skimage.feature import hog
import numpy as np
import abc

class _Descriptor(object):

    __metaclass__ = abc.ABCMeta
    
    def __init_(self, params):
        pass
    
    @abc.abstractmethod
    def get_features(self, images):
        pass

    @abc.abstractmethod
    def get_params(self):
        pass

class HogDesc(_Descriptor):

    def __init__(self, orientations=9, pix_per_cell=8, cell_per_block=2):
        self._orientations = orientations
        self._pix_per_cell = pix_per_cell
        self._cell_per_block = cell_per_block

    def get_features(self, images, feature_vector=True):
        """
        # Args
            images : ndarray, shape of (N, n_rows, n_cols)
                gray scale images
    
        # Returns
            features :
                if feature_vector == True:
                    2d-array, shape of (N, (pix_per_cell - cell_per_block + 1)**2 * cell_per_block**2 * orientations)
                else: 
                    5d-array, shape of (N,
                                       pix_per_cell - cell_per_block + 1,
                                       pix_per_cell - cell_per_block + 1,
                                       cell_per_block,
                                       cell_per_block,
                                       orientations)
        """

        features = []
        for img in images:
            feature_array = hog(img,
                                orientations=self._orientations,
                                pixels_per_cell=(self._pix_per_cell, self._pix_per_cell),
                                cells_per_block=(self._cell_per_block, self._cell_per_block),
                                visualise=False,
                                feature_vector=feature_vector)
            features.append(feature_array)
    
        features = np.array(features)
        return features
                
    def get_params(self):
        params = {"orientations" : self._orientations,
                  "pix_per_cell" : self._pix_per_cell,
                  "cell_per_block" : self._cell_per_block}
        return params



