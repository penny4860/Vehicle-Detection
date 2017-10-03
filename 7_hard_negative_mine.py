
import cv2
import matplotlib.pyplot as plt
import os

from car.detect import ImgDetector
from car.train import load_model
from car.data import list_files, files_to_images

import random

def get_hard_negative_imgs(img_detector, src_dir, dst_dir, n_samples=10000):
    """
    # Args
        img_detector : ImgDetector instance
    """

    negative_files = list_files(src_dir, pattern="*.jpg", random_order=False)
    random.shuffle(negative_files)

    cnt = 0
    for filename in negative_files:
        img = plt.imread(filename)
        img_detector.run(img, start_pt=(0,0), do_heat_map=False)
                
        for box in img_detector.detect_boxes:
            cnt += 1

            x1, y1, x2, y2 = box
            patch = img[y1:y2, x1:x2]
            patch = cv2.resize(patch, (64,64))

            fname = "{}.png".format(cnt)
            path = os.path.join(dst_dir, fname)
            plt.imsave(path, patch)
            print("{} is saved".format(fname))
            
            if cnt >= n_samples:
                break

if __name__ == "__main__":

    detector = ImgDetector(classifier=load_model("model.pkl"))
    get_hard_negative_imgs(detector,
                           "dataset//negative_imgs",
                           "dataset//non-vehicles//samples",
                           n_samples=20000)


