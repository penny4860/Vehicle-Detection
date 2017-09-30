
import os
from car.data import list_files, files_to_images, FileHDF5

CAR_DIR = "dataset//vehicles"
NON_CAR_DIR = "dataset//non-vehicles"

def get_features(images, n_orientations=9, pix_per_cell=8, cell_per_block=2):
    from skimage.feature import hog
    import numpy as np
    features = []
    for img in images:
        feature_array = hog(img,
                            orientations=n_orientations,
                            pixels_per_cell=(pix_per_cell, pix_per_cell),
                            cells_per_block=(cell_per_block, cell_per_block),
                            visualise=False,
                            feature_vector=True)
        features.append(feature_array)

    features = np.array(features)
    return features


if __name__ == "__main__":
    # 1. load image files
    car_files = list_files(CAR_DIR, pattern="*.png", random_order=False)
    negative_files = list_files(NON_CAR_DIR, pattern="*.png", random_order=False)
    print(len(car_files), len(negative_files))

    # 2. to ndarray
    pos_imgs = files_to_images(car_files)
    neg_imgs = files_to_images(negative_files)
    print(pos_imgs.shape, neg_imgs.shape)
    
    pos_features = get_features(pos_imgs)
    neg_features = get_features(neg_imgs)
    
    FileHDF5.write(pos_features, os.path.join(os.path.dirname(__file__), "car_db.hdf5"), "pos_features", "w")
    FileHDF5.write(neg_features, os.path.join(os.path.dirname(__file__), "car_db.hdf5"), "neg_features", "a")
    print(pos_features.shape, neg_features.shape)

