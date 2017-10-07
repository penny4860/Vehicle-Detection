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

입력 image에서 HOG feature vector 추출하여 sliding window 로 vehicle 영역을 scan하였습니다. scan이 끝난 후에는 heat map operation을 통해서 false positive patch 를 제거 하였고, 겹치는 positive patch 를 merge 하였습니다.

![alt text][image_framework]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [youtube link](https://www.youtube.com/watch?v=DgNtyNuCMbQ&feature=youtu.be) and [github link](https://github.com/penny4860/Vehicle-Detection/project_video_result.mp4)



####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

아래의 그림과 같이 heat map 을 구성하여 false positive patch 를 제거하고, overlap 영역에 대해서 merge 를 수행하였습니다.

![alt text][heat_framework]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##### 1) Limiatations of HOG + SVM classifier

저는 이 프로젝트에서 HOG feature extractor 와 SVM 을 이용한 classifier 를 구현하였습니다. 이러한 방식은 적은 수의 training sample 로도 괜찮은 수준의 classifier 를 구현할 수 있다는 점에서 장점이 있습니다. 
그러나, 이러한 방법은 CNN (Convolutional Neural Network) 를 사용한 방법보다 성능이 떨어 집니다. 만약에 training sample 을 더 수집할 수 있다면, CNN의 사용을 생각해 볼 수 있을 것입니다.

##### 2) Limitations of Sliding window fashion

Sliding window 방식은 이미지의 여러 patch에 classifier를 적용해서 각 patch를 desired object 또는 배경으로 분류합니다. 이러한 방식은 object detection 알고리즘을 구현할 때 가장 먼저 생각할 수 있는 방법이고, 성능도 나쁘지 않습니다. 그러나, slidinw window 방식은 2개 이상의 object가 근접해 있을 경우 이를 분리하는 것이 어렵습니다. 

![alt text][separation]

위 그림에서와 같이 still image 정보만으로는 근접해 있는 2개의 object를 분리해서 인식하는 것이 어려웠습니다. 저는 본 프로젝트에서 이 문제를 해결하기 위해 이전 frames 에서의 인식된 정보를 사용하였습니다. 
[Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)을 참고해서 kalman filter를 이용한 tracking 알고리즘을 구현했습니다. 그 결과 위 그림에서의 초록색 box와 같은 검출 결과가 도출되었습니다.

그러나, time-series information을 사용해서 still image에서의 detection 성능을 극복하는 것에는 한계가 있습니다. 
still image에서 인식 성능을 높이기 위해서는 [YOLO 9000](https://arxiv.org/abs/1612.08242) 이나 [SSD](https://arxiv.org/abs/1512.02325)와 같은 방법을 사용할 수 있습니다. 
이 논문들에서는 image 를 여러개의 작은 grid로 나누고, 각 grid 마다 여러개의 object를 검출하는 방식을 사용했습니다. 매우 효과적인 방법이라고 생각합니다.

