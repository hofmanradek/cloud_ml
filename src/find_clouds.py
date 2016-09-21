# -*- coding: utf-8 -*-
__author__ = 'radek hofman'

import sys
import rgb2hsi
import cv2
import numpy
import math
import scipy.io
import os


#paths
RESULTS_DIR = os.path.join("..","results")

#some useful constants
PI2 = 2*math.pi
LPI2 = - 0.5*math.log(PI2)

#shothand for log
log = math.log


def get_prior(HSI, img_mask=None, eps=255.):
    """
    calculates significance matrix
    """

    H = HSI[:,:,0]
    I = HSI[:,:,2]

    W = (I+eps)/(H+eps)
    W = rgb2hsi.map_to_n(W, 255).astype("uint8")
    o_thr, W_thr = cv2.threshold(W, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15, 15))
    eroded_thr = cv2.erode(W_thr, morph_kernel, iterations = 1)
    prior = cv2.dilate(eroded_thr, morph_kernel, iterations = 2)

    #masking of prior with and image mask
    if img_mask is None:
        img_mask = numpy.ones(prior.shape)

    #total pixel count
    pxcount = float(img_mask[img_mask > 0].shape[0])

    #calculate prior probabilities of  classes
    #prior probability of cloud
    pc = prior[numpy.logical_and(img_mask > 0, prior > 0)].shape[0] / pxcount
    #prior probability of nocloud
    pnc = prior[numpy.logical_and(img_mask > 0, prior == 0)].shape[0] / pxcount

    return prior, pc, pnc


def get_image_mask(RGB):
    """
    return binary mask of just image without black borders
    """

    gray = cv2.cvtColor(RGB, cv2.COLOR_RGB2GRAY)
    ret, img_mask = cv2.threshold(gray,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #Otsu's thresholding followed by a massive opening
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(101, 101))

    img_mask = cv2.erode(img_mask, morph_kernel, iterations = 3)
    img_mask = cv2.dilate(img_mask, morph_kernel, iterations = 3)
    img_mask = 1-img_mask

    return img_mask.astype("uint8")


def log_naive_bayes_matrix(features, cm, cv, ncm, ncv, pc=0.5, pnc=0.5):
    """
    - matrix version of naive Bayesian classifier for a set of features,
    two classes (pc, pnc) and a corresponding first and second moments
    of classes

    - priors pc and pnc have default uninformative values 0.5

    parameters:
    features: assumed features of all pixels in image
    cm: mean for class 'cloud'
    cv: covariance matrix for class 'cloud'
    ncm: mean for class 'nocloud'
    ncv: covariance matrix for class 'nocloud'
    pc:  prior for class 'cloud'
    pnc:  prior for class 'nocloud'

    naive Bayes => only diagonal elements of cov. matrices are used

    returns: 0 if p(nocloud|x) > p(cloud|x), 1 if p(cloud|x) > p(nocloud|x)
    """

    clogp = numpy.zeros(features[0].shape)  # cloud class log probability
    nclogp = numpy.zeros(features[0].shape)  # nocloud class log probability

    #this should work thanks to numpy broadcasting
    for ci in range(len(features)):
        print "......processing feature %d" % ci
        fea = features[ci]
        clogp += - math.log(math.sqrt(cv[ci, ci])) + LPI2 -0.5*(fea - cm[ci])**2/cv[ci, ci]
        nclogp += - math.log(math.sqrt(ncv[ci, ci])) + LPI2 -0.5*(fea - ncm[ci])**2/ncv[ci, ci]

    #finally, add prior
    clogp += log(pc)
    nclogp += log(pnc)

    return numpy.where(clogp > nclogp, 1, 0)


def classify(features, trained_data, pc, pnc):
    """
    loads trained class statistics and runs classification
    """

    try:
        dc = scipy.io.loadmat(trained_data)
        cm = dc["cloud_m"][0]
        cv = dc["cloud_cov"]
        ncm = dc["nocloud_m"][0]
        ncv = dc["nocloud_cov"]

    except IOError, e:
        print e
        print "Have you trained you classifier? Run train.py if not."
        sys.exit(1)

    return log_naive_bayes_matrix(features, cm, cv, ncm, ncv, pc=pc, pnc=pnc)


def eval_performance(data_dir, image, masks, img_mask=None):
    """
    - prints statistics of classification, uses comparison agains
     provided annotations


    error metrics = (false_negative + false_positive)/pixels_count
    """

    #remove file extension
    fname = "".join(file.split(".")[:-1])
    #reading ground truth - annotations
    gt = cv2.imread(os.path.join(data_dir, fname+"_clouds.png"), 0)

    #masking of prior with and image mask
    if img_mask is None:
        img_mask = numpy.ones(gt.shape)

    img_px_count = img_mask[img_mask>0].shape[0]

    classifiers = ("H/I-S-I", "RGB", "combined")

    print "...Classification performance:"
    for mi, mask in enumerate(masks):
        fn = numpy.sum(numpy.logical_and(numpy.logical_and(mask == 0, img_mask > 0), gt > 0))
        fp = numpy.sum(numpy.logical_and(mask > 0, numpy.logical_and(gt == 0, img_mask > 0)))
        print "......classifier %d - %9s: Classification error: %7.6f%% " % (mi, classifiers[mi], float(fp+fn)/img_px_count*100.)
    print "-------------------------------------"

def postprocess(cmhsi, cmrgb):
    """
    results postprocessing - noise filtering
    """

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    morph_kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11, 11))

    cmhsi = cv2.erode(cmhsi, morph_kernel, iterations = 1)
    cmhsi = cv2.dilate(cmhsi, morph_kernel, iterations = 1)

    cmrgb = cv2.erode(cmrgb, morph_kernel_1, iterations = 4)
    cmrgb = cv2.dilate(cmrgb, morph_kernel_1, iterations = 4)

    cloud_mask = cmhsi + cmrgb
    #getting back to 0-1
    cloud_mask = numpy.where(cloud_mask > 0, 1, 0)

    return cloud_mask, cmhsi, cmrgb


def classify_main(data_dir, image):
    """
    quote: 'Function takinga satellite scene (.tif) as input and returning mask
            of probability values for each pixel (1 means being classified as
            cloud, 0 means being classified as non-cloud)'
    """

    print "\nFile %s" % os.path.join(data_dir, image)
    print "...Converting image from RGB to HSI"
    HSI, RGB = rgb2hsi.HSI_matrix(os.path.join(data_dir, image))

    print "...Calculating approximate image mask"
    #uncomment to generate masks on the fly - it si slow:(
    #img_mask = get_image_mask(RGB)
    #currently we use precomputed masks in ../data/img_masks
    img_mask = scipy.io.loadmat(os.path.join("..","data","img_masks","mask_"+image))["image_mask"]

    print "...Calculating prior"
    prior, pc, pnc = get_prior(HSI, img_mask)
    print "...Prior cloud = %3.2f, non-cloud = %3.2f" % (pc, pnc)

    print "...Classifying"
    eps = 1.
    features = [(HSI[:,:,2]+eps)/(HSI[:,:,0]+eps), HSI[:,:,1], HSI[:,:,2]]
    trained_data = "trained_class_ihsi"
    cloud_mask_hsi = classify(features, trained_data, pc, pnc)
    #scipy.io.savemat("class_res", {"class_res": cloud_mask}, do_compression=True)

    print "...Classifying"
    features = [RGB[:,:,0], RGB[:,:,1], RGB[:,:,2]]
    trained_data = "trained_class_rgb"
    cloud_mask_rgb = classify(features, trained_data, pc, pnc)
    #scipy.io.savemat("class_res", {"class_res": cloud_mask}, do_compression=True)

    print "...Postprocessing results"
    cloud_mask, cloud_mask_hsi, cloud_mask_rgb = postprocess(cloud_mask_hsi.astype("uint8"),
                                                             cloud_mask_rgb.astype("uint8"))

    #saving results to RESULTS_DIR
    print "...Saving results to %s" % RESULTS_DIR
    filename = "cloud_mask_hsi_%s.png" % image
    cv2.imwrite(os.path.join(RESULTS_DIR, filename), cloud_mask_hsi*255)
    filename = "cloud_mask_rgb_%s.png" % image
    cv2.imwrite(os.path.join(RESULTS_DIR, filename), cloud_mask_rgb*255)
    filename = "cloud_mask_rgb+hsi_%s.png" % image
    cv2.imwrite(os.path.join(RESULTS_DIR, filename), cloud_mask*255)

    #prints statistics of classification
    eval_performance(data_dir, image, (cloud_mask_hsi, cloud_mask_rgb, cloud_mask))

    return cloud_mask


if __name__ == "__main__":

    #test files
    files = ("20140926_022050_090b_visual.tif",
             "20150805_024524_0906_visual.tif",
             "20150917_020427_1_0b0b_visual.tif",
             "20150922_234147_0c07_visual.tif")

    data_dir = os.path.join("..","data")
    for file in files:
        cloud_mask = classify_main(data_dir, file)