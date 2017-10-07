# Vehicle Detection and Tracking
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This is Udacity Self-Driving CarND Term 1 Project 5: Video for vehicle detection and tracking. 
I implemented Vehicle Detection and Tracking algorithm using HOG, SVM, and Kalman filter.

## Detection and Tracking Process

### 1. HOG feature extraction

### 2. SVM classfication

The trained classifier is stored at [model_v4.pkl](https://github.com/penny4860/Vehicle-Detection/blob/master/model_v4.pkl) as a binary file.

### 3. Sliding window scanning

<img src="output_images/scan.gif">

The source code for the multiple scale sliding window is here[scan module](https://github.com/penny4860/Vehicle-Detection/blob/master/car/scan.py).

## 4. Heatmap operation

<img src="output_images/heatmap.png">

## 5. Tracking using kalman filter

### 1) Object located in proximity

<img src="output_images/separation.png">

### 2) Obscured object tracking

<img src="output_images/tracking.png">

I implemented the tracking algorithm using the kalman filter by referring to [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763). As a result, the green box in the figure above was detected.


## Result Video

[Youtube Link](https://www.youtube.com/watch?v=DgNtyNuCMbQ)


## Report

[writeup](https://github.com/penny4860/Vehicle-Detection/writeup.md)

