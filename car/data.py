# -*- coding: utf-8 -*-

import os
import glob
import random

import cv2
import numpy as np
import h5py
from sklearn.utils import shuffle


class FileHDF5(object):
    @staticmethod
    def read(filename, db_name):
        db = h5py.File(filename, "r")
        np_data = np.array(db[db_name])
        db.close()
        
        return np_data

    @staticmethod
    def write(data, filename, db_name, write_mode="w"):
        """Write data to hdf5 format.

        # Args
            data : ndarray
            filename : str
                including full path
            db_name : str
                database name
        """
        def _check_directory(filename):
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.mkdir(directory)

        _check_directory(filename)        
        # todo : overwrite check
        db = h5py.File(filename, write_mode)
        dataset = db.create_dataset(db_name, data.shape, dtype="float")
        dataset[:] = data[:]
        db.close()


def list_files(directory, pattern="*.*", n_files_to_sample=None, recursive_option=True, random_order=True):
    """list files in a directory matched in defined pattern.

    # Args
        directory : str
            filename of json file
        pattern : str
            regular expression for file matching
        
        n_files_to_sample : int or None
            number of files to sample randomly and return.
            If this parameter is None, function returns every files.
        
        recursive_option : boolean
            option for searching subdirectories. If this option is True, 
            function searches all subdirectories recursively.

    # Returns
        conf : dict
            dictionary containing contents of json file
    """

    if recursive_option == True:
        dirs = [path for path, _, _ in os.walk(directory)]
    else:
        dirs = [directory]
    
    files = []
    for dir_ in dirs:
        for p in glob.glob(os.path.join(dir_, pattern)):
            files.append(p)
    
    if n_files_to_sample is not None:
        if random_order:
            files = random.sample(files, n_files_to_sample)
        else:
            files = files[:n_files_to_sample]
    return files


def files_to_images(files):
    images = []
    for filename in files:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)
    images = np.array(images)
    return images


def create_xy(pos_features, neg_features):
    pos_ys = np.ones((len(pos_features)))
    neg_ys = np.zeros((len(neg_features)))
    xs = np.concatenate([pos_features, neg_features], axis=0)
    ys = np.concatenate([pos_ys, neg_ys], axis=0)
    xs, ys = shuffle(xs, ys, random_state=0)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test
