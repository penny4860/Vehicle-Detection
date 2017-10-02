
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


class HogMap(object):
    
    def __init__(self, hog_desc=HogDesc()):
        self._img = None
        self._desc = hog_desc

    def _to_feature_map_point(self, x, y):
        """
        # Args
            x : start x point in image
            y : start y point in image

        # Returns
            x1 : x1 point in hog feature map
            y1 : y1 point in hog feature map
            x2
            y2
        """
        unit_dim = self._desc._pix_per_cell - self._desc._cell_per_block + 1
        x1 = x // self._desc._pix_per_cell
        y1 = y // self._desc._pix_per_cell
        x2 = x1 + unit_dim
        y2 = y1 + unit_dim
        return x1, y1, x2, y2

    def set_features(self, gray):
        self._img = gray
        self._feature_map = self._desc.get_features([gray], feature_vector=False)
    
    def get_features(self, x, y):
        x1, y1, x2, y2 = self._to_feature_map_point(x, y)
        feature_vector = self._feature_map[:, y1:y2, x1:x2, :, :, :].ravel().reshape(1, -1)
        return feature_vector



