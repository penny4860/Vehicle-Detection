
import os
from car.data import list_files, files_to_images, FileHDF5
from car.desc import get_hog_features

CAR_DIR = "dataset//vehicles"
NON_CAR_DIR = "dataset//non-vehicles"


if __name__ == "__main__":

    # 1. load features    
    pos_features = FileHDF5.read("car_db.hdf5", "pos_features")[:100]
    neg_features = FileHDF5.read("car_db.hdf5", "neg_features")[:100]
    print(pos_features.shape, neg_features.shape)

    # 2. create (X, y)     
    import numpy as np
    from sklearn.utils import shuffle
    pos_ys = np.ones((len(pos_features)))
    neg_ys = np.zeros((len(neg_features)))
    xs = np.concatenate([pos_features, neg_features], axis=0)
    ys = np.concatenate([pos_ys, neg_ys], axis=0)
    xs, ys = shuffle(xs, ys, random_state=0)
    print(xs.shape, ys.shape)

#     from sklearn.svm import SVC
#     from sklearn.model_selection import GridSearchCV
#     parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#     svr = SVC()
#     clf = GridSearchCV(svr, parameters)
#     clf.fit(xs, ys)
#     # clf.best_params_


