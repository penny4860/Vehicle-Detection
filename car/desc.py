
from skimage.feature import hog
import numpy as np

def get_hog_features(images, n_orientations=9, pix_per_cell=8, cell_per_block=2, feature_vector=True):
    """
    # Args
        images : ndarray, shape of (N, n_rows, n_cols)
            gray scale images

    # Returns
        features : ndarray, shape of (N, n_feature_vector)
    """
    features = []
    for img in images:
        feature_array = hog(img,
                            orientations=n_orientations,
                            pixels_per_cell=(pix_per_cell, pix_per_cell),
                            cells_per_block=(cell_per_block, cell_per_block),
                            visualise=False,
                            feature_vector=feature_vector)
        features.append(feature_array)

    features = np.array(features)
    return features
