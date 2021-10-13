# -*- coding: utf-8 -*-
import warnings
import argparse

import keras

from keras.layers import *
from keras.layers import Add
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, Conv2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.optimizers import *
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers.advanced_activations import PReLU
import keras.backend as K
import numpy as np
import cv2
import math
import pywt
import sys
import os
from functools import partial
from sklearn.model_selection import train_test_split
import dtcwt

def seperateChannel(input):
############ Seperate R, G, B kernel ############

    R_kernel = np.array([[[1, 0, 0]]])
    G_kernel = np.array([[[0, 1, 0]]])
    B_kernel = np.array([[[0, 0, 1]]])
    R = Conv2D(1, (1, 1), padding='same', kernel_initializer=keras.initializers.Constant(value=R_kernel),name="seperateR")(input)
    G = Conv2D(1, (1, 1), padding='same', kernel_initializer=keras.initializers.Constant(value=G_kernel),name="seperateG")(input)
    B = Conv2D(1, (1, 1), padding='same', kernel_initializer=keras.initializers.Constant(value=B_kernel),name="seperateB")(input)

    return R,G,B


class dtcwtLayer(Layer):
    def __init__(self, **kwargs):
        super(dtcwtLayer, self).__init__(**kwargs)
        self.__name__ = 'dtcwtLayer'

    def call(self, input):
        xfm = dtcwt.tf.Transform2d()    
        x = xfm.forward_channels(input, data_format = 'nhwc',nlevels=2)
        output_low = x.lowpass_op

        high_0_list = []
        for is_imag in range(2):
            for i in range(3):
                for j in range(6):
                    if is_imag == 0:
                        high_0_list.append(K.expand_dims(tf.math.real(x.highpasses_ops[0][:,:,:,i,j]),3))
                    elif is_imag == 1:
                        high_0_list.append(K.expand_dims(tf.math.imag(x.highpasses_ops[0][:,:,:,i,j]),3))
                    else:
                        raise ValueError
        output_high_0 = K.concatenate(high_0_list,axis=3)
        high_1_list = []
        for is_imag in range(2):
            for j in range(6):
                for i in range(3):
                    if is_imag == 0:
                        high_1_list.append(K.expand_dims(tf.math.real(x.highpasses_ops[1][:,:,:,i,j]),3))
                    elif is_imag == 1:
                        high_1_list.append(K.expand_dims(tf.math.imag(x.highpasses_ops[1][:,:,:,i,j]),3))
                    else:
                        raise ValueError
        output_high_1 = K.concatenate(high_1_list,axis=3)
        return [output_low, output_high_0, output_high_1]
    
    def get_config(self):
        base_config = super(dtcwtLayer, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        out_shape=(input_shape[0],input_shape[1]//2,input_shape[2]//2,input_shape[3])
        out_shape_h0=(input_shape[0],input_shape[1]//2,input_shape[2]//2,input_shape[3] * 12)
        out_shape_h1=(input_shape[0],input_shape[1]//4,input_shape[2]//4,input_shape[3] * 12)
        return [out_shape, out_shape_h0, out_shape_h1]


class inversedtcwtLayer(Layer):
    def __init__(self, **kwargs):
        super(inversedtcwtLayer, self).__init__(**kwargs)
        self.__name__ = 'inversedtcwtLayer'

    def call(self, input):
        xfm = dtcwt.tf.Transform2d()    
        #Input shape should be [0] (Low): 240*320*3; [1] (high[0]) 240*320*36; [2](high[1]) 120*160*36
        low_part = input[0]
        high_part0 = input[1]
        high_part1 = input[2]
        # high part 0 should be convert to 240*320*3*6 (complex)
        # high part input channel: Real(:,:,0,0->:,:,1,0->...->:,:,0,1->...)
        #                        , Imag(:,:,0,0->:,:,1,0->...->:,:,0,1->...)

        final_high_0_real_list = []
        for j in range(6):
            rgb_channel_list = []
            for i in range(3):
                rgb_channel_list.append(K.expand_dims(high_part0[:,:,:,i + 3 * j],3))
            rgb_channel = K.concatenate(rgb_channel_list,axis=3)
            final_high_0_real_list.append(K.expand_dims(rgb_channel,4))
        
        final_high_0_real = K.concatenate(final_high_0_real_list,axis=4)

        final_high_0_imag_list = []
        for j in range(6):
            rgb_channel_list = []
            for i in range(3):
                rgb_channel_list.append(K.expand_dims(high_part0[:,:,:,i + 3 * j + 18],3))
            rgb_channel = K.concatenate(rgb_channel_list,axis=3)
            final_high_0_imag_list.append(K.expand_dims(rgb_channel,4))
        
        final_high_0_imag = K.concatenate(final_high_0_imag_list,axis=4)
        final_high_0 = tf.complex(final_high_0_real, final_high_0_imag)

        final_high_1_real_list = []
        for j in range(6):
            rgb_channel_list = []
            for i in range(3):
                rgb_channel_list.append(K.expand_dims(high_part1[:,:,:,i + 3 * j],3))
            rgb_channel = K.concatenate(rgb_channel_list,axis=3)
            final_high_1_real_list.append(K.expand_dims(rgb_channel,4))
        
        final_high_1_real = K.concatenate(final_high_1_real_list,axis=4)

        final_high_1_imag_list = []
        for j in range(6):
            rgb_channel_list = []
            for i in range(3):
                rgb_channel_list.append(K.expand_dims(high_part1[:,:,:,i + 3 * j + 18],3))
            rgb_channel = K.concatenate(rgb_channel_list,axis=3)
            final_high_1_imag_list.append(K.expand_dims(rgb_channel,4))
        final_high_1_imag = K.concatenate(final_high_1_imag_list,axis=4)
        final_high_1 = tf.complex(final_high_1_real, final_high_1_imag)

        pyramid = dtcwt.tf.Pyramid(low_part, (final_high_0, final_high_1))
        inverse_result = xfm.inverse_channels(pyramid, data_format = 'nhwc')
        
        return inverse_result
    
    def get_config(self):
        base_config = super(inversedtcwtLayer, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        out_shape=(input_shape[0][0],input_shape[0][1]*2,input_shape[0][2]*2,input_shape[0][3])
        return out_shape

class sliceLayer(Layer):
    def __init__(self, edge, **kwargs):
        super(sliceLayer, self).__init__(**kwargs)
        self.edge = edge
        self.__name__ = 'sliceLayer'

    def call(self, input):
        s=K.int_shape(input)
        out=input[:,self.edge:s[1]-self.edge,self.edge:s[2]-self.edge,:]
        return out
    
    def get_config(self):
        config = {'edge': int(self.edge)}
        base_config = super(sliceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        out_shape=(input_shape[0],input_shape[1]-(2*self.edge),input_shape[2]-(2*self.edge),input_shape[3])
        return out_shape

def refine_H(input,inputD,initializer,nameAdjust, dropout=0.0):
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    L1 = Conv2D(64, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(inputD)
    L2 = Conv2D(64, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(inputD)
    L3 = Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(inputD)
    L = Concatenate()([L1,L2,L3])
    L = BatchNormalization(axis=channel_axis)(L)
    L = Dropout(dropout)(L)
    L_m = LeakyReLU(alpha=0.5)(L)
    
    L1 = Conv2D(64, (2, 2), padding='same', kernel_initializer = initializer)(L_m)
    L2 = Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(L_m)
    L3 = Conv2D(64, (5, 5), padding='same', kernel_initializer = initializer)(L_m)
    L4 = Conv2D(64, (9, 9), padding='same', kernel_initializer = initializer)(L_m)
    L1 = LeakyReLU(alpha=0.5)(L1)
    L2 = LeakyReLU(alpha=0.5)(L2)
    L3 = LeakyReLU(alpha=0.5)(L3)
    L4 = LeakyReLU(alpha=0.5)(L4)
    L_c = Concatenate()([L1,L2,L3,L4])
    L_c = BatchNormalization(axis=channel_axis)(L_c)
    L_c = Dropout(dropout)(L_c)
    L_c = LeakyReLU(alpha=0.5)(L_c)
    
    x1 = Conv2D(54, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x2 = Conv2D(54, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x3 = Conv2D(54, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x = Concatenate()([x1,x2,x3])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Dropout(dropout)(x)
    x_m = LeakyReLU(alpha=0.5)(x)
    
    x1 = Conv2D(18, (2, 2), padding='same', kernel_initializer = initializer)(x_m)
    x2 = Conv2D(18, (3, 3), padding='same', kernel_initializer = initializer)(x_m)
    x3 = Conv2D(18, (5, 5), padding='same', kernel_initializer = initializer)(x_m)
    x1 = LeakyReLU(alpha=0.5)(x1)
    x2 = LeakyReLU(alpha=0.5)(x2)
    x3 = LeakyReLU(alpha=0.5)(x3)
    x_c = Concatenate()([x1,x2,x3])
    x_c = BatchNormalization(axis=channel_axis)(x_c)
    x_c = Dropout(dropout)(x_c)
    x_c = LeakyReLU(alpha=0.5)(x_c)
    x_L = Concatenate()([x_c,L_c])
    
    x1 = Conv2D(48, (2, 2), padding='same', kernel_initializer = initializer)(x_L)
    x2 = Conv2D(48, (3, 3), padding='same', kernel_initializer = initializer)(x_L)
    x3 = Conv2D(48, (5, 5), padding='same', kernel_initializer = initializer)(x_L)
    x1 = LeakyReLU(alpha=0.5)(x1)
    x2 = LeakyReLU(alpha=0.5)(x2)
    x3 = LeakyReLU(alpha=0.5)(x3)
    x_c = Concatenate()([x1,x2,x3])
    x_c = BatchNormalization(axis=channel_axis)(x_c)
    x_c = Dropout(dropout)(x_c)
    
    deconvATT_1 = Conv2DTranspose(32, (2, 2), padding='same', strides = 2, kernel_initializer = initializer, name='deconvATT_1'+nameAdjust)(x_c)
    deconvATT_2 = Conv2DTranspose(32, (3, 3), padding='same', strides = 2, kernel_initializer = initializer, name='deconvATT_2'+nameAdjust)(x_c)
    deconvATT_3 = Conv2DTranspose(32, (7, 7), padding='same', strides = 2, kernel_initializer = initializer, name='deconvATT_3'+nameAdjust)(x_c)
    deconvATT_1 = LeakyReLU(alpha=0.5)(deconvATT_1)
    deconvATT_2 = LeakyReLU(alpha=0.5)(deconvATT_2)
    deconvATT_3 = LeakyReLU(alpha=0.5)(deconvATT_3)
    mergeX = Concatenate()([deconvATT_1, deconvATT_2, deconvATT_3])
    
    conv1_1 = Conv2D(64, (15, 1), padding='same', kernel_initializer = initializer)(mergeX)
    conv1_2 = Conv2D(64, (1, 15), padding='same', kernel_initializer = initializer)(mergeX)
    conv2_1 = Conv2D(18, (1, 15), padding='same', kernel_initializer = initializer)(conv1_1)
    conv2_2 = Conv2D(18, (15, 1), padding='same', kernel_initializer = initializer)(conv1_2)
    GCN = Concatenate()([conv2_1, conv2_2])
    
    conv1 = Conv2D(36, (3, 3), padding='same', kernel_initializer = initializer)(GCN)
    conv1 = LeakyReLU(alpha=0.5)(conv1)
    conv2 = Conv2D(36, (3, 3), padding='same', kernel_initializer = initializer)(conv1)
    
    c1 = Conv2D(36, (3, 3), padding='same', kernel_initializer = initializer)(conv2)
    c1 = LeakyReLU(alpha=0.5)(c1)
    c2 = Conv2D(36, (3, 3), padding='same', kernel_initializer = initializer)(c1)
    BRatt = add([c2, conv2])
    BRatt = Conv2D(36, (3, 3), padding='same', kernel_initializer = initializer)(BRatt)
    output = PReLU()(BRatt)
    
    m = add([init, output])
    return m
    
def refine_L(input,inputD,initializer,nameAdjust, dropout=0.0):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    #First Part
    L1 = Conv2D(64, (7, 7), padding='same', kernel_initializer = initializer)(inputD)
    L2 = Conv2D(64, (7, 7), padding='same', kernel_initializer = initializer)(input)
    L = Concatenate()([L1,L2])
    L_m = LeakyReLU(alpha=0.3)(L)

    #Second Part
    #Init
    x_0 = Conv2D(64, (7, 7), padding='same', strides=(2, 2), kernel_initializer = initializer)(L_m)

    #Block 1
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(x_0)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x_1 = add([x, x_0])

    #Block 2
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(x_1)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x_2 = add([x, x_1])

    #Block 3
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(x_2)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x_3 = add([x, x_2])

    #Block 4
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(x_3)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x_4 = add([x, x_3])

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(x_4)
    x = LeakyReLU(alpha=0.3)(x)
    deconv = Conv2DTranspose(64, (3, 3), padding='same', strides = 2, kernel_initializer = initializer, name='deconv_1'+nameAdjust)(x)
    upsamp = UpSampling2D((2, 2))(x)
    final = Concatenate()([deconv,upsamp])
    final = add([final, L_m])
    output = Conv2D(3, (3, 3), padding='same', kernel_initializer = initializer)(final)
    return output

def DTCWT_Model(inp,gt,postFix):


    output_0_low, output_0_h0, output_0_h1  = dtcwtLayer()(inp)
    output_1_low, output_1_h0, output_1_h1  = dtcwtLayer()(output_0_low)

    #GT

    output_gt_0_low, output_gt_0_h0, output_gt_0_h1  = dtcwtLayer()(gt)
    output_gt_1_low, output_gt_1_h0, output_gt_1_h1  = dtcwtLayer()(output_gt_0_low)
    
    #downfromInp=AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(inp)
    
    
    initializer='he_normal'
    
    D1 = Conv2D(12, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(inp)
    D2 = Conv2D(12, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(inp)
    D3 = Conv2D(12, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(inp)
    
    downfromInp = Concatenate()([D1,D2,D3])

    D1 = Conv2D(12, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(output_0_low)
    D2 = Conv2D(12, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(output_0_low)
    D3 = Conv2D(12, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(output_0_low)
    
    downfromx1L = Concatenate()([D1,D2,D3])
    
    D1 = Conv2D(12, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(downfromInp)
    D2 = Conv2D(12, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(downfromInp)
    D3 = Conv2D(12, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(downfromInp)
    
    downfromInp2 = Concatenate()([D1,D2,D3])

    D1 = Conv2D(12, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(downfromInp2)
    D2 = Conv2D(12, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(downfromInp2)
    D3 = Conv2D(12, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(downfromInp2)
    
    downfromInp3 = Concatenate()([D1,D2,D3])

    
    downConcat2_L = Concatenate()([downfromInp2, downfromx1L])
    
    x2LR = refine_L(output_1_low,downConcat2_L,'he_normal',postFix+'L_2', dropout=0.3)
    
    downConcat2 = Concatenate()([x2LR, downfromInp2, downfromx1L])
    
    x2H0R = refine_H(output_1_h0,downConcat2,'he_normal',postFix+'H1_2', dropout=0.2)
    x2H1R = refine_H(output_1_h1,downfromInp3,'he_normal',postFix+'H2_2', dropout=0.2)

    x_IDWT2 = inversedtcwtLayer()([x2LR, x2H0R, x2H1R])

    downConcat_L = Concatenate()([x_IDWT2,downfromInp])
    
    x1LR = refine_L(output_0_low,downConcat_L,'he_normal',postFix+'L_1', dropout=0.3)
    
    downConcat = Concatenate()([x1LR,downfromInp])
    
    x1H0R = refine_H(output_0_h0,downConcat,'he_normal',postFix+'H1_1', dropout=0.2)
    x1H1R = refine_H(output_0_h1,downfromInp2,'he_normal',postFix+'H2_1', dropout=0.2)

    x_IDWT3 = inversedtcwtLayer()([x1LR, x1H0R, x1H1R])

    gtDwtList=[output_gt_0_h0, output_gt_0_h1, output_gt_1_h0, output_1_h1, output_gt_0_low, output_gt_1_low]
    RefineDwtList=[x1H0R, x1H1R, x2H0R, x2H1R, x1LR, x2LR]
    
    return x_IDWT3,gtDwtList,RefineDwtList

def build_model(shape):

    input = Input(shape=shape)
    inputGT = Input(shape=shape)

    x_R ,gtListR ,RefineListR = DTCWT_Model(input,inputGT,'noSep')

    x = x_R
    
    output = sliceLayer(edge=16,name='Output')(x)
    outputBCP = Activation('linear',name='CCPLoss')(output)
    
    model = Model([input,inputGT],[x, output, outputBCP], name='DTCWT_DESNOW')
    
    partial_dwt_lossFinal = partial(partial_dwt_loss,refine_pred=[RefineListR],refine_gt=[gtListR])
    partial_dwt_lossFinal.__name__ = 'partial_dwt_loss' # Keras requires function names
    
    return model,partial_dwt_lossFinal

def partial_dwt_loss(y_true, y_pred,refine_pred,refine_gt):
    
    if len(refine_pred[0]) != len(refine_gt[0]):
      print('Failure!  len(refine_pred[0] != len(refine_gt[0]')
      assert len(refine_pred[0]) == len(refine_gt[0])
    
    mseFinal = 0
    
    for i in range(len(refine_gt[0])):
        mseFinal+=L2_Charbonnier_loss(refine_gt[0][i],refine_pred[0][i])
        
    
    return mseFinal

def L1_Charbonnier_loss(y_true, y_pred):
    diff = K.sqrt( (y_true-y_pred)**2 + 1e-6 )
    loss = K.mean(diff) #K.sum(diff)
    return loss
    
def L2_Charbonnier_loss(y_true, y_pred):
    diff = K.sqrt( (y_true-y_pred)**2 + 1e-6 ) * K.sqrt( (y_true-y_pred)**2 + 1e-6 )
    loss = K.mean(diff) #K.sum(diff)
    return loss

def ccp_loss(y_true, y_pred):
    #change pool size to change the patch size
    poolsize = 35
    y_pred = K.min(y_pred, axis = 3, keepdims = True)
    y_pred = K.pool2d(y_pred, pool_size=(poolsize,poolsize), strides=(1,1),
                          padding='same', data_format=None,
                          pool_mode='max')
    
    y_true = K.min(y_true, axis = 3, keepdims = True)
    y_true = K.pool2d(y_true, pool_size=(poolsize,poolsize), strides=(1,1),
                          padding='same', data_format=None,
                          pool_mode='max')

    return K.sigmoid(K.mean(K.abs(y_pred - y_true)))
 
def VGGloss(y_true, y_pred):  # Note the parameter order
    from keras.applications.vgg16 import VGG16
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input
    
    vggmodel = VGG16(include_top=False, weights='imagenet')
    vggmodel.trainable=False
    f_p = vggmodel(y_pred)  
    f_t = vggmodel(y_true)  
    return K.mean(K.square(f_p - f_t))
   
def generate_data_generator(datagenerator, X, X2, Y1,BATCHSIZE,seed=1):
    genX1 = datagenerator.flow(X,batch_size = BATCHSIZE, seed=seed)
    genX2 = datagenerator.flow(X2,batch_size = BATCHSIZE, seed=seed)
    genY1 = datagenerator.flow(Y1,batch_size = BATCHSIZE, seed=seed)
    genY2 = datagenerator.flow(Y1,batch_size = BATCHSIZE, seed=seed)
    while True:
            Xi1 = genX1.next()/255
            Xi2 = genX2.next()/255
            Yi1 = genY1.next()/255
            Yi2 = genY2.next()/255
            Yi3 = Yi2 + 0
            yield [Xi1, Xi2], [Yi1, Yi2, Yi3]
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a dtcwt desnow model.')
    parser.add_argument('--logPath', type=str)
    parser.add_argument('--dataPath', type=str, default='/path_to_data/data.npy')
    parser.add_argument('--gtPath', type=str, default='/path_to_gt/gt.npy')
    parser.add_argument('--batchsize', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--modelPath', type=str, default='', help='Model Path if con train')
    parser.add_argument('--validation_num', type=int, default=200, help='Number of validation image')
    parser.add_argument('--steps_per_epoch', type=int, default=80, help='steps_per_epoch')
    parser.add_argument('--note', type=str, default='', help='Add whatever you want' )
    
    
    args = parser.parse_args()

    print("Parameters:")
    print('Log Path:',args.logPath)
    print('Data Path:',args.dataPath)
    print('Gt Path:',args.gtPath)
    print('batchsize:',args.batchsize)
    print('epochs:',args.epochs)
    print('modelPath:',args.modelPath)

    model,partial_dwt_lossFinal = build_model((512,672,3))
    
    if args.modelPath!= '':
        print("Continue train!")
        model.load_weights(args.modelPath,by_name=True)
        print('Load Weights Success!')
    
    
    opt = Adam(lr=0.0001)
    model.compile(optimizer = opt, loss = [partial_dwt_lossFinal, VGGloss, ccp_loss],
        loss_weights = [2,0.1,2]
      )
    
    if not os.path.exists(args.logPath):
        os.mkdir(args.logPath)
    
    #Copy the input param to log
    print('Save input param to logPath...')
    print('Argv len:',len(sys.argv))
    f = open(args.logPath + '/inputParam.txt','w')
    for i in range(len(sys.argv)):
        f.write(sys.argv[i]+' ')
        print('Write:',sys.argv[i])
    f.close()
    
    print('load Data')
    data=np.load(args.dataPath)
    print('load Gt')
    label=np.load(args.gtPath)
    
    print('Transform data and label to ycb_cr')
    for i in range(data.shape[0]):
      data[i]=cv2.cvtColor(data[i],cv2.COLOR_BGR2YCR_CB)
      label[i]=cv2.cvtColor(label[i],cv2.COLOR_BGR2YCR_CB)
    
    print('Convert End!')
    
    print('random select 100 pic for testing')
    data, val_data, label, val_label= train_test_split(data, label, test_size=args.validation_num,shuffle=True)
    
    print(data.shape,'data shape')
    print(label.shape,'label shape')
    print(val_data.shape,'val_data shape')
    print(val_label.shape,'val_label shape')
    print(data[0][0][0],'data 0 0 0')
    print(np.max(data),np.min(data),'max min data')
    
    data=np.pad(data,((0,0),(16,16),(16,16),(0,0)),'constant')
    val_data=np.pad(val_data,((0,0),(16,16),(16,16),(0,0)),'constant')
    labelPad=np.pad(label,((0,0),(16,16),(16,16),(0,0)),'constant')
    val_labelPad=np.pad(val_label,((0,0),(16,16),(16,16),(0,0)),'constant')
    
    print('Img Gene')
    image_datagen = ImageDataGenerator(featurewise_center=False,
                         featurewise_std_normalization=False,
                         rotation_range=20,
                         width_shift_range=0.15,
                         height_shift_range=0.15,
                         zoom_range=0.10)
    seed = 1
            
    val_data_gen = ImageDataGenerator(featurewise_center=False,
                         featurewise_std_normalization=False)
    
    print('Batchsize=',args.batchsize)
    print('Epochs=',args.epochs)
    
    print(data.shape,'data shape')
    print(label.shape,'Gt shape')
    CSVlogName=args.logPath+'/log.csv'
    checkPointPath=args.logPath+'/model.{epoch:04d}-{val_loss:.4f}.h5'
    callback=[
            TensorBoard(log_dir=args.logPath),
            CSVLogger(CSVlogName,append=True),
            ModelCheckpoint(checkPointPath, period=100),
            ModelCheckpoint(args.logPath+'/modelBest.h5', monitor='val_loss', verbose=1,
                                save_best_only=True, mode='min', save_weights_only=False)
            
        ]
    
    history = model.fit_generator(
                generate_data_generator(image_datagen, data,labelPad, label,args.batchsize,1),    
                epochs = args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                validation_data=generate_data_generator(val_data_gen,val_data, val_labelPad,val_label,args.batchsize,1),
                validation_steps=args.validation_num/args.batchsize,
                callbacks=callback
            )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
