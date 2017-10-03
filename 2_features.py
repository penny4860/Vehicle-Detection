
import os
from car.data import list_files, files_to_images, FileHDF5
from car.desc import HogDesc

CAR_DIR = "dataset//vehicles"
NON_CAR_DIR = "dataset//non-vehicles"
HARD_NEGATIVE_DIR = "dataset//hard-negatives"


if __name__ == "__main__":
#     # 1. load image files
#     car_files = list_files(CAR_DIR, pattern="*.png", random_order=False)
#     negative_files = list_files(NON_CAR_DIR, pattern="*.png", random_order=False)
#     print(len(car_files), len(negative_files))
# 
#     # 2. to ndarray
#     pos_imgs = files_to_images(car_files)
#     neg_imgs = files_to_images(negative_files)
#     print(pos_imgs.shape, neg_imgs.shape)
#     
#     pos_features = get_hog_features(pos_imgs)
#     neg_features = get_hog_features(neg_imgs)
#     
#     FileHDF5.write(pos_features, os.path.join(os.path.dirname(__file__), "car_db.hdf5"), "pos_features", "w")
#     FileHDF5.write(neg_features, os.path.join(os.path.dirname(__file__), "car_db.hdf5"), "neg_features", "a")
#     print(pos_features.shape, neg_features.shape)


    hard_negative_files = list_files(HARD_NEGATIVE_DIR, pattern="*.png", random_order=False)
    hnm_imgs = files_to_images(hard_negative_files)
    desc = HogDesc()
    hnm_features = desc.get_features(hnm_imgs)
    print(hnm_features.shape)
    FileHDF5.write(hnm_features, os.path.join(os.path.dirname(__file__), "car_db.hdf5"), "hnm_features", "a")

    


