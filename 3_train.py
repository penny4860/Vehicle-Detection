
import os
from car.data import list_files, files_to_images, FileHDF5
from car.desc import get_hog_features

CAR_DIR = "dataset//vehicles"
NON_CAR_DIR = "dataset//non-vehicles"

def evaluate_params(X, y, X_test, y_test, params):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    clf = GridSearchCV(SVC(), params)
    clf.fit(X, y)

    print("1. Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("2. Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("    %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))


def create_xy(pos_features, neg_features):
    import numpy as np
    from sklearn.utils import shuffle
    pos_ys = np.ones((len(pos_features)))
    neg_ys = np.zeros((len(neg_features)))
    xs = np.concatenate([pos_features, neg_features], axis=0)
    ys = np.concatenate([pos_ys, neg_ys], axis=0)
    xs, ys = shuffle(xs, ys, random_state=0)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    # 1. load features    
    pos_features = FileHDF5.read("car_db.hdf5", "pos_features")
    neg_features = FileHDF5.read("car_db.hdf5", "neg_features")

    # 2. create (X, y)     
    X_train, X_test, y_train, y_test = create_xy(pos_features, neg_features)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1], 'C': [10]}]

    # {'kernel': 'rbf', 'gamma': 0.1, 'C': 10}
    evaluate_params(X_train, y_train, X_test, y_test, tuned_parameters)
