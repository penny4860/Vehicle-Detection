# -*- coding: utf-8 -*-

import cv2
from car.train import load_model
from car.detect import ImgDetector

from car.data import list_files
import os
import random
def get_patch_samples(src_dir, dst_dir):
    img_files = list_files(src_dir)

    cnt = 1
    for fname in img_files:
        img = cv2.imread(fname)
        d = ImgDetector(classifier=load_model("model_v3.pkl"))
        _ = d.run(img, do_heat_map=True)
    
        boxes = d.detect_boxes + d.heat_boxes
        for box in boxes:
            x1, y1, x2, y2 = box
            
            for _ in range(4):
                offset_x = random.randint(-8, 8)
                offset_y = random.randint(-8, 8)
                
                xx1 = offset_x + x1
                xx2 = offset_x + x2
                yy1 = offset_y + y1
                yy2 = offset_y + y2
                
                patch = img[yy1:yy2, xx1:xx2, :]
                if patch.shape[0] > 0 and patch.shape[1] > 0:
                    patch = cv2.resize(patch, (64,64))
                    cv2.imwrite(os.path.join(dst_dir, "{}.png".format(cnt)), patch)
                    print(os.path.join(dst_dir, "{}.png".format(cnt)))
                    cnt += 1

if __name__ == "__main__":
    get_patch_samples("video", "samples")
