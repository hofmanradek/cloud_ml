# -*- coding: utf-8 -*-
__author__ = 'radek hofman'

import rgb2hsi

import sys
import cv2
import numpy
import math
import scipy.io
import os
import abc
import logging
logging.basicConfig(level=logging.DEBUG)  # filename='classify.log'

#some useful constants
PI2 = 2*math.pi
LPI2 = - 0.5*math.log(PI2)

#shothand for log
log = math.log


class Classifier(object):

    def __init__(self, class_type, train_data_path):
        self.class_type = class_type
        self.load_trained_data(train_data_path)

    def load_trained_data(self, train_data_path):
        try:
            dc = scipy.io.loadmat(train_data_path)
            self.cm = dc["cloud_m"][0]
            self.cv = dc["cloud_cov"]
            self.ncm = dc["nocloud_m"][0]
            self.ncv = dc["nocloud_cov"]
        except IOError, e:
            logging.error(e)
            logging.error('Have you trained you classifier? Run train.py if not.')
            sys.exit(1)

    @abc.abstractmethod
    def extract_features(self, HSI, RGB):
        pass

    @staticmethod
    def compute_prior(HSI, img_mask=None, eps=1., ses=1):
        """
        calculates significance matrix
        this is present in both RGB and HSI since it calculates an estimate of prior
        """
        H = HSI[:,:,0]
        I = HSI[:,:,2]

        W = (I+eps)/(H+eps)
        W = rgb2hsi.map_to_n(W, 255).astype("uint8")
        o_thr, W_thr = cv2.threshold(W, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ses, ses))
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

    @staticmethod
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
            logging.debug("......processing feature %d" % ci)
            fea = features[ci]
            clogp += - math.log(math.sqrt(cv[ci, ci])) + LPI2 -0.5*(fea - cm[ci])**2/cv[ci, ci]
            nclogp += - math.log(math.sqrt(ncv[ci, ci])) + LPI2 -0.5*(fea - ncm[ci])**2/ncv[ci, ci]

        #finally, add prior
        clogp += log(pc)
        nclogp += log(pnc)

        return numpy.where(clogp > nclogp, 1, 0)

    @staticmethod
    def postprocess(cloud_mask, ses=1):
        """
        results postprocessing - noise filtering
        """
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ses, ses))
        cloud_mask = cv2.erode(cloud_mask, morph_kernel, iterations = 1)
        cloud_mask = cv2.dilate(cloud_mask, morph_kernel, iterations = 1)
        #transform back to 0-1
        cloud_mask = numpy.where(cloud_mask > 0, 1, 0)
        return cloud_mask

    def classify(self, file_path):  #, data_dir, filename):
        logging.info("Runnign %s classifier on %s" % (self.class_type, file_path))
        HSI, RGB = rgb2hsi.HSI_matrix(file_path)
        prior, pc, pnc = self.compute_prior(HSI)
        features = self.extract_features(HSI, RGB)
        cloud_mask = self.log_naive_bayes_matrix(features, self.cm, self.cv, self.ncm, self.ncv, pc=pc, pnc=pnc)
        cloud_mask = self.postprocess(cloud_mask)
        return cloud_mask

    @staticmethod
    def eval_performance(gt_path, masks, img_mask=None):
        """
        - prints statistics of classification, uses comparison against
         a binary mask

        error metrics = (false_negative + false_positive)/pixels_count
        """
        #remove file extension
        fname = "".join(file.split(".")[:-1])
        #reading ground truth - annotations
        gt = cv2.imread(gt_path, 0)

        #masking of prior with and image mask
        if img_mask is None:
            img_mask = numpy.ones(gt.shape)

        img_px_count = img_mask[img_mask>0].shape[0]

        classifiers = ("H/I-S-I", "RGB", "combined")

        logging.info("...Classification performance:")
        for mi, mask in enumerate(masks):
            fn = numpy.sum(numpy.logical_and(numpy.logical_and(mask == 0, img_mask > 0), gt > 0))
            fp = numpy.sum(numpy.logical_and(mask > 0, numpy.logical_and(gt == 0, img_mask > 0)))
            logging.info("......classifier %d - %9s: Classification error: %7.6f%% " % (mi, classifiers[mi], float(fp+fn)/img_px_count*100.))
        logging.info("-------------------------------------")


class ClassifyRGB(Classifier):
    """
    classifier in [R,G,B] space
    """
    def extract_features(self, HSI, RGB):
        R, G, B = RGB[:,:,0], RGB[:,:,1], RGB[:,:,2]
        features = [R, G, B]
        return features


class ClassifyHSI(Classifier):
    """
    Classifier in [(I+eps)/(H+eps),S,I] space
    """
    def __init__(self, eps, *args):
        """
        this classifier has an additional parameter epsilon, which can be set
        """
        self.eps = eps
        super(self.__class__, self).__init__(*args)

    def extract_features(self, HSI, RGB):
        H, S, I = HSI[:,:,0], HSI[:,:,1], HSI[:,:,2]
        features = [(I+self.eps)/(H+self.eps), S, I]
        return features


if __name__ == "__main__":

    #test files - let's find clouds in them! wooohooooo!
    #all test files are courtesy of NASA or NOAA
    files = ("clouds1.jpg",
             "clouds2.jpg",
             "clouds3.jpg",
             "clouds4.jpg",
             "clouds5.jpg",
             "clouds6.jpg")

    test_data_dir = os.path.join("..","data","test")
    train_data_dir = os.path.join("..","data","train")

    #classifiers
    crgb = ClassifyRGB('rgb', os.path.join(train_data_dir, 'trained_class_rgb.mat'))
    chsi = ClassifyHSI(1., 'hsi', os.path.join(train_data_dir, 'trained_class_hsi.mat'))

    for image in files:
        file_path = os.path.join(test_data_dir, image)
        mask_rgb = crgb.classify(file_path)
        mask_hsi = chsi.classify(file_path)
        #we add results of our two classifiers
        cloud_mask = numpy.where(mask_rgb + mask_hsi > 0, 1, 0)

        #saving results to test_data_dir
        logging.info("...Saving results to %s" % test_data_dir)
        filename = "%s_cloud_mask_hsi.png" % image
        cv2.imwrite(os.path.join(test_data_dir, filename), mask_hsi*255)
        filename = "%s_cloud_mask_rgb.png" % image
        cv2.imwrite(os.path.join(test_data_dir, filename), mask_rgb*255)
        filename = "%s_cloud_mask_rgb+hsi.png" % image
        cv2.imwrite(os.path.join(test_data_dir, filename), cloud_mask*255)
