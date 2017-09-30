

from sklearn.svm import SVC
from car.data import FileHDF5, create_xy
from car.train import evaluate_params


if __name__ == "__main__":

    # 1. load features    
    pos_features = FileHDF5.read("car_db.hdf5", "pos_features")
    neg_features = FileHDF5.read("car_db.hdf5", "neg_features")

    # 2. create (X, y)     
    X_train, X_test, y_train, y_test = create_xy(pos_features, neg_features)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1], 'C': [10]}]

    # {'kernel': 'rbf', 'gamma': 0.1, 'C': 10}
    # evaluate_params(X_train, y_train, X_test, y_test, tuned_parameters)

    clf = SVC(C=10.0, kernel='rbf', gamma=1.0)
    clf.fit(X_train, y_train)

    from sklearn.metrics import classification_report
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    from car.train import save_model
    save_model(clf, "model.pkl")

