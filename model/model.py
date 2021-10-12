# -*- coding: utf-8 -*-
import warnings
import argparse
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import Model
from keras.models import load_model
import keras.backend as K
import numpy as np
import cv2
import math
import pywt
import sys
import dtcwt

print('import end')

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
        low_part = input[0]
        high_part0 = input[1]
        high_part1 = input[2]
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

class bound_relu(Layer):
    def __init__(self, maxvalue, thres = 0, **kwargs):
        super(bound_relu, self).__init__(**kwargs)
        self.maxvalue = K.cast_to_floatx(maxvalue)
        self.thres = K.cast_to_floatx(thres)
        self.__name__ = 'bound_relu'

    def call(self, inputs):
        return keras.activations.relu(inputs, max_value=self.maxvalue, threshold = self.thres)

    def get_config(self):
        config = {'maxvalue': float(self.maxvalue),'thres': float(self.thres)}
        base_config = super(bound_relu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

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
 
def DTCWT_Model(inp, postFix):


    output_0_low, output_0_h0, output_0_h1  = dtcwtLayer()(inp)
    output_1_low, output_1_h0, output_1_h1  = dtcwtLayer()(output_0_low)
    
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

    RefineDwtList=[x1H0R, x1H1R, x2H0R, x2H1R, x1LR, x2LR]
    
    return x_IDWT3, RefineDwtList

def build_DTCWT_model(shape):

    input = Input(shape=shape)
    x_R ,RefineListR = DTCWT_Model(input,'noSep')
    x = x_R
    output = sliceLayer(edge=16,name='FinalOutput')(x)
    model = Model([input],[output], name='DTCWT_Model')
    return model
