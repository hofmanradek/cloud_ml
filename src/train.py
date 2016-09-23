# -*- coding: utf-8 -*-
__author__ = 'radek hofman'

import numpy
import cv2
import scipy.io
import os
import rgb2hsi


def train_classifier(data_dir, files):
    """
    Trains Bayesian classifier with normal pdfs from images in files
    """
    cloud_stack = []
    nocloud_stack = []

    for i, file in enumerate(files):
        print "Training on file %s" % file

        #remove file extension
        fname = "".join(file.split(".")[:-1])
        #load ground truth - provided annotations
        gt = cv2.imread(os.path.join(data_dir, fname+"_mask.png"), 0)

        #read HSL and RGB channels
        HSI, RGB = rgb2hsi.HSI_matrix(os.path.join(data_dir, file))
        H, S, I = HSI[:,:,0], HSI[:,:,1], HSI[:,:,2]
        R, G, B = RGB[:,:,0], RGB[:,:,1], RGB[:,:,2]

        #defining feature vectors
        eps = 1.
        features = [(I+eps)/(H+eps), S, I] #this has better component separability than HSL
        #features = [R, G, B]

        #training follows
        cloud_1 = []
        nocloud_1 = []

        for nf, fea in enumerate(features):
            f_cloud = fea[gt > 0]
            f_nocloud = fea[gt == 0]  # no cloud without borders

            cloud_1.append(numpy.array(f_cloud))
            nocloud_1.append(numpy.array(f_nocloud))

        cloud_stack.append(numpy.vstack(cloud_1))
        nocloud_stack.append(numpy.vstack(nocloud_1))

    cloud = numpy.hstack(cloud_stack)
    nocloud = numpy.hstack(nocloud_stack)

    cloud_m = numpy.ma.masked_invalid(cloud.mean(axis=1))
    cloud_cov = numpy.cov(cloud)

    nocloud_m = nocloud.mean(axis=1)
    nocloud_cov = numpy.cov(nocloud)


    print "Training statistics:"
    print "Cloud class mean:", cloud_m
    print "Cloud class cov. mat.:", cloud_cov
    print "Nocloud class mean:", nocloud_m
    print "Nocloud class cov. mat.:", nocloud_cov

    scipy.io.savemat(os.path.join(data_dir, "trained_class_ihsi"), {"cloud_m": cloud_m, "cloud_cov": cloud_cov,
                                       "nocloud_m": nocloud_m, "nocloud_cov": nocloud_cov})


if __name__ == "__main__":
    #specify data dir
    data_dir = os.path.join("..","data","train")

    #specify which files use for classifier training
    files = ("clouds1.jpg",
             "clouds2.jpg",
             "clouds3.jpg",
             "clouds4.jpg")

    #FEATURE VECTORS MUST BE SET INSIDE train_classifier()!!!
    train_classifier(data_dir, files)
