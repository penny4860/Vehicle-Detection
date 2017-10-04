

from sklearn.svm import SVC
from car.data import FileHDF5, create_xy
from car.train import test
import numpy as np


if __name__ == "__main__":

    # 1. load features    
    pos_features = FileHDF5.read("car_db.hdf5", "pos_features")
    neg_features = FileHDF5.read("car_db.hdf5", "neg_features")
    hnm_features = FileHDF5.read("car_db.hdf5", "hnm_features")
    ext_pos_features = FileHDF5.read("car_db.hdf5", "ext_pos_features")
    ext_neg_features = FileHDF5.read("car_db.hdf5", "ext_neg_features")
    # print(pos_features.shape, neg_features.shape, hnm_features.shape, ext_pos_features.shape, ext_neg_features.shape)
    
    pos_features = np.concatenate([pos_features, ext_pos_features], axis=0)
    neg_features = np.concatenate([neg_features, hnm_features, ext_neg_features], axis=0)
    print(pos_features.shape, neg_features.shape)
 
    # 2. create (X, y)     
    X_train, X_test, y_train, y_test = create_xy(pos_features, neg_features)
      
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1], 'C': [10]}]
   
    # {'kernel': 'rbf', 'gamma': 0.1, 'C': 10}
    # evaluate_params(X_train, y_train, X_test, y_test, tuned_parameters)
   
    clf = SVC(C=0.15, kernel='rbf', gamma=1.0, class_weight={0:1.0, 1:7.0})
    clf.fit(X_train, y_train)
   
    test(clf, X_train, y_train)
    test(clf, X_test, y_test)
      
    from car.train import save_model
    save_model(clf, "model_v4.pkl")
#         1.0       0.95      0.97      0.96       981       1:5
#         1.0       0.94      0.98      0.96       981       1:5
#         1.0       0.93      0.98      0.96       981        1:10




# (9627, 1764) (19670, 1764)
#              precision    recall  f1-score   support
# 
#         0.0       1.00      0.99      1.00     15704
#         1.0       0.99      1.00      0.99      7733
# 
# avg / total       1.00      1.00      1.00     23437
# 
#              precision    recall  f1-score   support
# 
#         0.0       0.99      0.98      0.99      3966
#         1.0       0.96      0.98      0.97      1894
# 
# avg / total       0.98      0.98      0.98      5860



