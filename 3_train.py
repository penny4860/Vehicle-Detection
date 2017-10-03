

from sklearn.svm import SVC
from car.data import FileHDF5, create_xy
import numpy as np
import random

import cv2
def random_crop(image, preserve_range=(12, 51)):
    print(image.shape)
    
    sx = random.randint(0, preserve_range[0]+1)
    sy = random.randint(0, preserve_range[0]+1)
    
    ex = random.randint(preserve_range[1], 64)
    ey = random.randint(preserve_range[1], 64)
    print(sx, sy, ex, ey)
    
    cropped = image[sy:ey, sx:ex]
    resized = cv2.resize(cropped, (64,64))
    return resized
    

if __name__ == "__main__":
    CAR_DIR = "dataset//vehicles"
    from car.data import list_files, files_to_images
    car_files = list_files(CAR_DIR, pattern="*.png", random_order=False)[:10]
    pos_imgs = files_to_images(car_files)
    from car.utils import plot_images
    plot_images([pos_imgs[0], random_crop(pos_imgs[0]), random_crop(pos_imgs[0]), random_crop(pos_imgs[0])])

#     # 1. load features    
#     pos_features = FileHDF5.read("car_db.hdf5", "pos_features")
#     neg_features = FileHDF5.read("car_db.hdf5", "neg_features")
#     hnm_features = FileHDF5.read("car_db.hdf5", "hnm_features")
#     neg_features = np.concatenate([neg_features, hnm_features], axis=0)
#     print(neg_features.shape)
# 
#     # 2. create (X, y)     
#     X_train, X_test, y_train, y_test = create_xy(pos_features, neg_features)
#      
#     # Set the parameters by cross-validation
#     tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1], 'C': [10]}]
#  
#     # {'kernel': 'rbf', 'gamma': 0.1, 'C': 10}
#     # evaluate_params(X_train, y_train, X_test, y_test, tuned_parameters)
#  
#     clf = SVC(C=10.0, kernel='rbf', gamma=1.0, class_weight='balanced')
#     clf.fit(X_train, y_train)
#  
#     from sklearn.metrics import classification_report
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#  
#     from car.train import save_model
#     save_model(clf, "model_hnm.pkl")

