##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image_framework]: ./output_images/image_framework.png
[heat_framework]: ./output_images/heatmap.png
[separation]: ./output_images/separation.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I implemented the code related to feature extraction in `car.desc.py module`.

* In training process, I extracted feature maps for sample images using HogDesc class.
* In the recognition process by the sliding window, the feature map is extracted in a way that only the HOG map for the entire image is obtained once in order to optimize the calculation time. I used the HogMap class here.

####2. Explain how you settled on your final choice of HOG parameters.

There are three tuning parameters of HOG.

* orientations
* pixels_per_cell
* cells_per_block

In a situation where the patch size has the same length and width, these three parameters determine the dimension of the feature vector with the following formula.

```
n_cells = patch_size / pixels_per_cell
n_features = (n_cells - cell_per_block + 1)**2 * cell_per_block**2 * orientations
```

According to the above equation, the parameters that have the greatest effect on feature dimention are `pix_per_cell` and` cell_per_block`.

To reduce the dimensionality of the feature vector, these two parameters used the smallest possible value to keep the discriminatory power. The rest of the parameter, `orientations`, uses the default value.

The final HOG parameters used are as follows.

* orientations : 9
* pixels_per_cell : 8
* cells_per_block : 2


Since the project uses a 64x64 image patch, the dimension size of the feature vector is 1764.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

##### 1) Kernel

I selected rbf kernel SVM from rbf kernel SVM and linear SVM. Experimental results show that the classification performance of rbf kernel SVM is better.

##### 2) Gamma

I chose `1.0` using the grid search function provided by scikit-learn.

##### 3) C

The `C` of the SVM determines the size of the margin, which has the greatest effect on the classification performance of unseen data.

I divided the dataset into train set and test set and changed the C value to monitor the classification performance.

To reduce the risk of overfitting, I tuned `C` to a smaller value and finally set the value of` C` to 0.15. (A small tuning of `C` has the effect of increasing the margin and reducing the test error.)

##### 4) Performance

The experimental results for the samples we have are shown below.

First, performance for trainig sample.

|Trainig Sample          | Precision     |  Recall       | F1-score      | Support       |  
|:----------------------:|:-------------:|:-------------:|:-------------:|:-------------:| 
| background sample      | 1.00          | 0.99          | 1.00          | 15704         |
| vehicle sample         | 0.99          | 1.00          | 0.99          | 7733          |
| avg / total            | 1.00          | 1.00          | 1.00          | 23437         |


Next, performance for the test sample.

|Test Sample             | Precision     |  Recall       | F1-score      | Support       |  
|:----------------------:|:-------------:|:-------------:|:-------------:|:-------------:| 
| background sample      | 0.99          | 0.98          | 0.99          | 3966         |
| vehicle sample         | 0.96          | 0.98          | 0.97          | 1894          |
| avg / total            | 0.98          | 0.98          | 0.98          | 5860         |


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The logic associated with the sliding window is implemented in the `car.scan.py` module. It is divided into the following three classes and implemented.

* ImgScanner: This class is responsible for sliding search on a single scale.
* ImgPyramid: This is a class that makes multiple scale pyramid for image.
* MultipleScanner: A wrapper class that has the above two class instances as member variables. From outside the module, create an instance of this class and perform a sliding window search on multiple scales.

There are two parameters related to the sliding window.

* search step: A step in which the window moves forward at a time. At test time, the HOG feature is obtained for the entire image rather than for each patch. Therefore, the size of one cell is set to 16 out of 8.
* scale: means the rate at which the layer shrinks when constructing an image pyramid. The smaller the value, the shorter the scan time, but the better the detection performance. We experimented with values ​​between 0.6 and 0.8 and set it to 0.8.

<img src="output_images/scan.gif">


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I extract the HOG feature vector from the input image and scan the vehicle area with a sliding window. After the scan, I removed the false positive patch through the heat map operation and merge the overlapping positive patches.

![alt text][image_framework]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [youtube link](https://www.youtube.com/watch?v=DgNtyNuCMbQ&feature=youtu.be) and [github link](https://github.com/penny4860/Vehicle-Detection/project_video_result.mp4)



####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As shown in the figure below, I constructed a heat map to remove false positive patches and perform a merge on the overlap area.

![alt text][heat_framework]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##### 1) Limiatations of HOG + SVM classifier

I implemented the HOG feature extractor and classifier using SVM in this project. This approach has the advantage of being able to implement decent classifiers with a small number of training samples.
However, this method has less performance than the CNN (Convolutional Neural Network) method. If you can collect more training samples, you might consider using CNN.

##### 2) Limitations of Sliding window fashion

Sliding window method classifies each patch into desired object or background by applying classifier to various patches of image. This is the first thing you can think of when implementing an object detection algorithm, and its performance is not bad. However, the slidinw window method is difficult to separate if two or more objects are close together.

![alt text][separation]

As shown in the figure above, it was difficult to separate two objects in close proximity by still image information. I used the perceived information from previous frames to solve this problem in this project. I implemented the tracking algorithm using the kalman filter by referring to [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763). As a result, the green box in the figure above was detected.

However, there is a limit to overcoming detection performance in still images using time-series information.
To improve recognition performance in still images, you can use methods such as [YOLO 9000](https://arxiv.org/abs/1612.08242) or [SSD](https://arxiv.org/abs/1512.02325).
The authors of this paper used a method of dividing the image into multiple small grids and detecting multiple objects for each grid. I think this is a very effective and interesting way.


