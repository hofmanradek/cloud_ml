# Classification of clouds in RGB satellite images #

## 1. Introduction
The goal is to develop an algorithm for clouds detection in colour satellite images. Source images are have 8-bit colour depth with just three channels R, G and B. These channels alone are usually not used to detect clouds. However, it is not a completely desperate situation. A clue could be the fact that clouds have usually higher intensity because of their higher reflexivity over non-cloud regions. One can train a classifier to detect these regions where all three channels are high. This works better for thick opaque clouds and worse for thinner semitransparent clouds (semitransparent clouds are there always because the are on the edges of the thick ones). To be able to detect also thinner clouds we follow approach of Zhang and Xiao (2014). Their approach is based on an observation made over hundreds of cloud images and says that when converted to HSI colour space, clouds have usually lower hues and higher intensities. In other words, cloud regions should have higher ratio I/H than non-cloud regions.
We face a two class problem which can be solved using a supervised or unsupervised classification method. Although we have just eight source images we decided to apply a supervised classification method. Four images with their annotations will be used for classifier training and the other four with their annotation will be used for evaluation of classification performance. We attempt to develop an algorithm combining results from two Bayesian classifiers operating on features {R, G, B} and {I/H, S, I} for detection of opaque and transparent clouds, respectively. Classification will be on a per pixel basis, i.e. without any context of surrounding pixels. As a prior for Bayesian classifier — which is very important for this class of classifiers — we use a mask obtained via thresholding I/H image in the sense of Zhang and Xiao (2014). This prior will be computed for each testing image and we believe that it will be better than general priors calculated over training set or uninformative uniform priors.

## 2. Algorithm

### 2.1 Image pre-processing

Firstly, we equalize histograms of RGB components using `cv2.equalizeHist()`. This gives us higher contrast and thus better class separation during classification. Then we convert RBG into HSI. Since this conversion is neither in OpenCV nor in `python.colorsys`, we decided to implement it using following formulas from (Pratt, 2001):

![Screen Shot 2016-09-24 at 21.30.21.png](https://bitbucket.org/repo/GKKr4n/images/4023270089-Screen%20Shot%202016-09-24%20at%2021.30.21.png)


Having HSI, we compute significance map W which should highlight the differences between cloud and non-cloud regions as follows

![Screen Shot 2016-09-24 at 21.30.38.png](https://bitbucket.org/repo/GKKr4n/images/2223698740-Screen%20Shot%202016-09-24%20at%2021.30.38.png)

Here, ε is an amplification factor set to 255. W is mapped on the range 0-255 and Otsu’s adaptive thresholding from OpenCV is applied to it (Otsu, 1975). We do not want to set the threshold manually to some best tuned value for two reasons. Firstly, we prefer objective methods with the lowest number of tunable parameters as possible; and secondly, we have just 4 images in our training set which is really poor number for doing such empirical conclusions. Thresholded image is then eroded and dilated (using morphological operation open from `OpenCV`) in order to remove small clusters and single pixels. Resulting binary image serves us as a prior for Bayesian classifier. Probabilities of classes are computed as a ratio of cloud or non-cloud pixels and the total pixel count (after masking with image mask segmenting images from their black borders, see Section 2.3 for details).

### 2.2 Classification

As a classifier we employ Bayesian classifier

![Screen Shot 2016-09-24 at 21.30.47.png](https://bitbucket.org/repo/GKKr4n/images/1236754141-Screen%20Shot%202016-09-24%20at%2021.30.47.png)

where both prior probability of a class p(c) and likelihood p(x|c) of a feature vector x given class c are normally distributed. Since we have full covariance matrices trained from millions of pixels, we could use multivariate Gaussian distributions but because it is slow, we use a simplification called naive Bayesian classifier where elements of a feature vector x are assumed conditionally independent given class c:

![Screen Shot 2016-09-24 at 21.30.59.png](https://bitbucket.org/repo/GKKr4n/images/2578716297-Screen%20Shot%202016-09-24%20at%2021.30.59.png)

Luckily, we do not have to calculate the denominator because it is just a function of x and it is same for all (two in our case) classes. We have to calculate only

![Screen Shot 2016-09-24 at 21.31.17.png](https://bitbucket.org/repo/GKKr4n/images/223684763-Screen%20Shot%202016-09-24%20at%2021.31.17.png)

where p(c) are given from preprocessing and p(xi|c) are calculated as normal likelihoods using mean and variance for a given feature in given class obtained from the training phase. ∝ is a sign for proportionality.

We implemented both naive (with conditional independence of features given a class) and a regular classifier operating with normal Gaussian distributions. For experiments we use the naive one because of its significantly higher speed. It is implemented without for loops and all operations are in a matrix fashion using Numpy broadcasting capabilities (http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html).

### 2.3 Classifier training

Class statistics were calculated from training set of four images for both RGB and (I/H)SI features and are stored in files `trained_class_rgh.mat` and `trained_class_hsi.mat`. In both cases, we estimated mean and full covariance matrices.

### 2.4 Combination of classifiers

The idea is that classification based on RGB channels detect more opaque clouds and that based on (I/H)SI detects more transparent clouds (we chose {I/H,S,I} over {H,S,I} because from training data it seems that {I/H, S, I} has better separability of classes). So, from these two classification results we create a single one as their union. Final mask will have probably some noise (small clouds, single pixels) which we remove using morphological opening.
Overview of the whole classification algorithm follows.

![Screen Shot 2016-09-24 at 21.45.19.png](https://bitbucket.org/repo/GKKr4n/images/1781124300-Screen%20Shot%202016-09-24%20at%2021.45.19.png)

### 2.5 Software


For most operations we use `OpenCV` library. Whenever possible we try to implement operations in a matrix fashion in order to use all the power on `Numpy`. During the whole computation we preserve 8-bit per channel representation stored as `uint8` type. For plotting images we use `Matplotlib`. For storing data (e.g. precomputed image masks) we use Matlab format written and read by Scipy.io module. Complete list of software with versions is attached to the source code in file `requirements.txt`. All paths are relative so it should work without any configuration.

