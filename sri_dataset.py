
import sys
import os
import logging

import numpy as np
from scipy.misc import imread

import PIL
from PIL import Image

from pylearn2.datasets import dense_design_matrix

_logger = logging.getLogger(__name__)

# inherited class to DenseDesignMatrix class (pylearn2)
class SRI_IR(dense_design_matrix.DenseDesignMatrix):
    
    """
    Loads SRI IR dataset: JKUNG
    """

    def __init__(self, which_set, start=None, stop=None):

        # set SRI_IR dataset parameters
        self.im_W, self.im_H = 128, 128
        self.img_shape = (1, self.im_W, self.im_H)
        self.img_size = np.prod(self.img_shape)
        self.label_names = ['positive', 'negative']
        self.n_classes = len(self.label_names)  # positive (1) or negative (0) 
                                                # on human segmentation in IR video frame
 
        # check which_set parameter 
        if which_set not in ['train', 'test']:
            if which_set == 'valid':
                raise ValueError(
                    "There is no SRI_IR validation set. SRI_IR dataset"
                    "consists of 256,351 train examples and 4,035 test"
                    "examples. If you need to use a validation set you"
                    "should divide the train set yourself.")
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

        # set data path before reading files
        data_path = '/media/deeplearn/DeepLearningDZ/peopleIR/CNN/'
        if which_set == 'train':
            fname = data_path + 'train0.txt'
        elif which_set == 'test':
            fname = data_path + 'test0.txt'

        # check file existence
        if not os.path.exists(fname):
            raise IOError(fname + " was not found."
                "The path or filename should be checked!")

        _logger.info('loading file %s' % fname)
        data_ptr = open(fname)

        # read data from the jpeg files
        X_list = []
        y_list = []
        for i,line in enumerate(data_ptr):
            [jpeg_fname,label] = line.split(' ')
            label = label.split('\n')[0]
            
            # read IR dataset
            if i >= start and i < stop:
                _logger.info('loading file %s' % jpeg_fname)
                rgb_im = Image.open(jpeg_fname)
                new_im = self.scale_image(rgb_im)
                imarray = np.asarray(new_im.getdata(0)).flatten('C')    # flattened array for a channel of 2D RGB image 
                                                                        # grey-level input has identical channels
                                                                        # C: row-major order, F: column-major order
                X_list.append(imarray)
                y_list.append(label)
            elif i == stop:
                break
        
        # convert list to numpy 'float32' array    
        self.X = np.cast['float32'](X_list)
        self.y = np.cast['uint8'](y_list)
 
    def scale_image(self,input_img):

        new_img = input_img.resize((self.im_W, self.im_H), PIL.Image.ANTIALIAS)

        return new_img

