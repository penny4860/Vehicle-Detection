

from sklearn.svm import SVC
from car.data import FileHDF5, create_xy
from car.train import test
import numpy as np


if __name__ == "__main__":

    # 1. load features    
    pos_features = FileHDF5.read("car_db.hdf5", "pos_features")
    neg_features = FileHDF5.read("car_db.hdf5", "neg_features")
    hnm_features = FileHDF5.read("car_db.hdf5", "hnm_features")
    neg_features = np.concatenate([neg_features, hnm_features], axis=0)
    print(neg_features.shape)

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
    save_model(clf, "model_v3.pkl")
#         1.0       0.95      0.97      0.96       981       1:5
#         1.0       0.94      0.98      0.96       981       1:5
#         1.0       0.93      0.98      0.96       981        1:10




#              precision    recall  f1-score   support
# 
#         0.0       1.00      1.00      1.00     15552
#         1.0       0.99      1.00      0.99      7003
# 
# avg / total       1.00      1.00      1.00     22555
# 
#              precision    recall  f1-score   support
# 
#         0.0       0.99      0.99      0.99      3850
#         1.0       0.97      0.97      0.97      1789
# 
# avg / total       0.98      0.98      0.98      5639

