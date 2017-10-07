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
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[image_framework]: ./output_images/image_framework.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

`car.desc.py module`에 feature extraction에 관련된 code를 구현하였습니다. 

* Training 과정에서는 HogDesc instance를 사용해서 sample images 에 대한 feature map을 추출하였습니다.
* Sliding window 에 의한 인식과정에서는 연산시간을 최적화 하기 위해서 image 전체에 대한 HOG map은 1번만 구하는 방식으로 feature map을 추출하였습니다. 여기서는 HogMap instance를 사용하였습니다.


####2. Explain how you settled on your final choice of HOG parameters.

HOG 의 tuning parameter는 다음의 3가지 입니다.

* orientations
* pixels_per_cell
* cells_per_block

patch size의 가로와 세로의 길이가 같은 상황에서, 이 3가지 parameter 에 의해서 다음의 수식으로 feature vector의 dimension이 결정됩니다.

```
n_cells = patch_size / pixels_per_cell
n_features = (n_cells - cell_per_block + 1)**2 * cell_per_block**2 * orientations
```

위 식에 의하면, feature dimention에 가장 큰 영향을 주는 parameter는 `pix_per_cell`과 `cell_per_block` 입니다.

feature vector의 dimension을 줄이기 위해 이 2개의 parameter는 분별력을 유지하는 한도내에서 가능한 최소의 값을 사용하였습니다. 나머지 parameter인 ```orientations``` 는 default 값을 사용하였습니다.

최종적으로 사용한 HOG parameter는 다음과 같습니다.

* orientations : 9
* pixels_per_cell : 8
* cells_per_block : 2

본 project에서는 64x64 size의 image patch를 사용하였기 때문에, feature vector의 dimension size는 1764가 되었습니다.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

##### 1) Kernel

rbf kernel SVM 과 linear SVM 중에서 rbf kernel SVM을 선택하였습니다. 실험결과 rbf kernel SVM의 분류성능이 더 좋았기 때문입니다.

##### 2) Gamma

scikit-learn에서 제공하는 grid search 함수를 사용해서 `1.0` 을 선택하였습니다.

##### 3) C

SVM 의 `C` 는 margin의 크기를 결정하기 때문에 unseen data의 분류성능에 가장 큰 영향을 줍니다.

dataset을 train set과 test set으로 나누고 C값을 바꿔가면서 분류성능을 monitoring 했습니다.

overfitting에 대한 위험성을 줄이기 위해 `C` 값을 작게 tuning하였고, 최종적으로 `C` 값을 0.15로 정하였습니다. (`C`값을 작게 tuning 하는 것이 margin을 크게 하는 효과가 있어, test error를 작게 하는 효과가 있습니다.)

##### 4) Performance

보유한 sample에 대한 실험결과는 아래와 같습니다.

먼저, trainig sample 에 대한 성능입니다.

|Trainig Sample          | Precision     |  Recall       | F1-score      | Support       |  
|:----------------------:|:-------------:|:-------------:|:-------------:|:-------------:| 
| background sample      | 1.00          | 0.99          | 1.00          | 15704         |
| vehicle sample         | 0.99          | 1.00          | 0.99          | 7733          |
| avg / total            | 1.00          | 1.00          | 1.00          | 23437         |


다음으로, test sample 에 대한 성능입니다.

|Test Sample             | Precision     |  Recall       | F1-score      | Support       |  
|:----------------------:|:-------------:|:-------------:|:-------------:|:-------------:| 
| background sample      | 0.99          | 0.98          | 0.99          | 3966         |
| vehicle sample         | 0.96          | 0.98          | 0.97          | 1894          |
| avg / total            | 0.98          | 0.98          | 0.98          | 5860         |


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

sliding window 와 관련된 logic은 `car.scan.py` module에 구현해 두었습니다. 다음의 3가지 class로 나누어서 구현하였습니다.

* ImgScanner : 단일 scale에 대해서 sliding search를 담당하는 class 입니다.
* ImgPyramid : image에 대해서 multiple scale pyramid를 구성하는 class 입니다.
* MultipleScanner : 위 2가지 class instance를 member 변수로 갖는 wrapper class 입니다. module의 외부에서는 이 class 의 instance를 생성해서 multiple scale에 대한 sliding window search를 수행합니다.

sliding window 와 관련된 parameter는 아래의 2가지가 있습니다.

* search step : window가 한번에 전진하는 step을 의미 합니다. test time에서는 HOG feature를 patch 별로 구하는 것이 아니라 image 전체에 대해서 구하게 됩니다. 따라서 1개 cell의 크기인 8의 배수중에서 16으로 정하였습니다.
* scale : image pyramid 를 구성할 때 layer가 줄어드는 비율을 의미합니다. 작은 값을 사용할 수록 scan 시간이 단축되는 효과가 있지만, detection 성능이 안좋아 질 수 있습니다. 0.6 ~ 0.8 사이의 값을 실험해 보았고, 0.8로 정하였습니다.

<img src="output_images/scan.gif">


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

입력 image에서 HOG feature vector 추출하여 sliding window 로 vehicle 영역을 scan하였습니다. scan이 끝난 후에는 heat map operation을 통해서 false negative patch 를 제거 하였고, 겹치는 positive patch 를 merge 하였습니다.

![alt text][image_framework]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_result.mp4)

youtube 링크 추가


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

