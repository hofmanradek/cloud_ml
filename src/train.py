# -*- coding: utf-8 -*-
__author__ = 'radek hofman'

import rgb2hsi

import numpy
import cv2
import scipy.io
import os
import abc
import logging
logging.basicConfig(level=logging.DEBUG)  # filename='train.log'


class TrainClassifier(object):
    """
    Trains Bayesian classifier with Normal pdfs from images in files
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, filenames):
        self.data_dir = data_dir
        self.filenames = filenames
        self.train()

    @abc.abstractmethod
    def extract_features(self):
        pass

    def train(self):
        cloud_stack = []
        nocloud_stack = []

        for i, filename in enumerate(self.filenames):
            logging.info('Training on file %s' % filename)

            #remove file extension
            fname = ''.join(filename.split('.')[:-1])
            #load ground truth - provided annotations
            gt = cv2.imread(os.path.join(data_dir, fname+'_mask.png'), 0)

            self.extract_features(filename)

            #training follows
            cloud_1 = []
            nocloud_1 = []

            for nf, fea in enumerate(self.features):
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

        self.print_stats(cloud_m, cloud_cov, nocloud_m, nocloud_cov)

        output_path = os.path.join(data_dir, 'trained_class_%s.mat' % self.class_type)
        logging.info("Saving output to %s" % output_path)
        scipy.io.savemat(output_path, {'cloud_m': cloud_m, 'cloud_cov': cloud_cov,
                                           'nocloud_m': nocloud_m, 'nocloud_cov': nocloud_cov})

    @staticmethod
    def print_stats(cloud_m, cloud_cov, nocloud_m, nocloud_cov):

        logging.debug('Training statistics:')
        logging.debug('Cloud class mean:\n%s' % str(cloud_m))
        logging.debug('Cloud class cov. mat.:\n%s' % str(cloud_cov))
        logging.debug('Nocloud class mean:\n%s' % str(nocloud_m))
        logging.debug('Nocloud class cov. mat.:\n%s' % str(nocloud_cov))


class TrainRGBClassifier(TrainClassifier):
    """
    RGB color space
    """
    def __init__(self, class_type, *args):
        self.class_type = class_type
        super(self.__class__, self).__init__(*args)

    def extract_features(self, filename):
        HSI, RGB = rgb2hsi.HSI_matrix(os.path.join(self.data_dir, filename))
        R, G, B = RGB[:,:,0], RGB[:,:,1], RGB[:,:,2]
        self.features = [R, G, B]


class TrainHSIClassifier(TrainClassifier):
    """
    HSI color space
    """
    def __init__(self, class_type, *args):
        self.class_type = class_type
        super(self.__class__, self).__init__(*args)

    def extract_features(self, filename):
        HSI, RGB = rgb2hsi.HSI_matrix(os.path.join(self.data_dir, filename))
        H, S, I = HSI[:,:,0], HSI[:,:,1], HSI[:,:,2]
        eps = 1.
        self.features = [(I+eps)/(H+eps), S, I] #this has better component separability than HSL


if __name__ == '__main__':
    """
    specify which files use for classifier training

      Data_dir must contain sattelite images with clouds and corresponding
      masks with the same filename and suffix _mask

      example: cloud1.png, cloud1_mask.png
    """

    data_dir = os.path.join('..', 'data', 'train')

    IMAGE_FORMATS = ('png', 'jpg', 'jpeg', 'tif', 'tiff')
    files = filter(lambda x: x.lower().split('.')[-1] in IMAGE_FORMATS and not '_mask' in x, os.listdir(data_dir))

    logging.debug('Found following files: %s' % str(files))

    TrainRGBClassifier('rgb', data_dir, files)
    TrainHSIClassifier('hsi', data_dir, files)