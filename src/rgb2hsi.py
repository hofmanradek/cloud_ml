# -*- coding: utf-8 -*-
__author__ = 'radek hofman'

import numpy
import math
import cv2
import random


#some constants
SQRT6 = -math.sqrt(6.)/6.
SQRT3 = -2.*SQRT6
PI = math.pi
PI2 = 2*math.pi
TH = 1./3.


def map_to_n(img,n):
    """
    maps an array img to a range 0-n
    """

    return (img - img.min()) / (img.max()-img.min()) * n


def HSI_matrix(image_path):
    """
    Conversion from RGB to HSI in a matrix form
    returns a HSI and RGB images per channel
    """

    def get_surr_val(n):
        return numpy.array([random.choice([-1., 1.]) for i in range(n)])

    #reads GBR image
    img = cv2.imread(image_path, 1)

    nx, ny = img.shape[:2]
    RGB = numpy.zeros(img.shape, dtype="uint8")
    HSI = numpy.zeros(img.shape, dtype="uint8")

    #histogram equalization
    R = cv2.equalizeHist(img[:,:,2])
    G = cv2.equalizeHist(img[:,:,1])
    B = cv2.equalizeHist(img[:,:,0])

    #flatting images into vectors for linear algebra reasons:)
    rgb_flat=numpy.vstack((numpy.ravel(R), numpy.ravel(G), numpy.ravel(B)))

    I  = numpy.dot(numpy.array([    TH,     TH,    TH]), rgb_flat)
    V1 = numpy.dot(numpy.array([ SQRT6,  SQRT6, SQRT3]), rgb_flat)
    V2 = numpy.dot(numpy.array([-SQRT6, -SQRT3,    0.]), rgb_flat)

    #instead of zeros we put uniformly symmetriall values -1 or 1
    V1[V1 == 0.] = get_surr_val(len(V1[V1 == 0.]))

    S = numpy.sqrt(numpy.power(V1, 2)+numpy.power(V2, 2))
    #H = numpy.where(V1 != 0., (numpy.arctan(V2/V1)+PI/2.)*255./PI, (PI/2.)*255./PI)
    H = (numpy.arctan(V2/V1)+PI/2.)*255./PI

    #casting to uint8
    I = numpy.reshape(I, (nx,ny)).astype("uint8")
    S = numpy.reshape(S, (nx,ny)).astype("uint8")
    H = numpy.reshape(H, (nx,ny)).astype("uint8")

    HSI[:,:,0] = H
    HSI[:,:,1] = S
    HSI[:,:,2] = I

    RGB[:,:,0] = R
    RGB[:,:,1] = G
    RGB[:,:,2] = B

    return HSI, RGB


if __name__ == "__main__":
    print "This is not intended for running."



