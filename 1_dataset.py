
import os
from car.data import list_files, files_to_images, FileHDF5

CAR_DIR = "dataset//vehicles"
NON_CAR_DIR = "dataset//non-vehicles"


if __name__ == "__main__":
    # 1. load image files
    car_files = list_files(CAR_DIR, pattern="*.png", random_order=False)
    negative_files = list_files(NON_CAR_DIR, pattern="*.png", random_order=False)
    print(len(car_files), len(negative_files))

    # 2. to ndarray
    pos_imgs = files_to_images(car_files)
    neg_imgs = files_to_images(negative_files)
    print(pos_imgs.shape, neg_imgs.shape)
    
    FileHDF5.write(pos_imgs, os.path.join(os.path.dirname(__file__), "car_db.hdf5"), "pos_imgs", "w")
    FileHDF5.write(neg_imgs, os.path.join(os.path.dirname(__file__), "car_db.hdf5"), "neg_imgs", "a")
    
#     img = pos_imgs[0]
#     from skimage.feature import hog
#     orient = 9
#     pix_per_cell = 8
#     cell_per_block = 2
#     feature_array = hog(img,
#                         orientations=orient,
#                         pixels_per_cell=(pix_per_cell, pix_per_cell),
#                         cells_per_block=(cell_per_block, cell_per_block),
#                         visualise=False,
#                         feature_vector=True)
#     print(feature_array.shape) # (7, 7, 2, 2, 9)

