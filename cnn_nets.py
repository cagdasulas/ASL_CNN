'''
this module involves the implementation of three different network structures:
    (1) U-Net
    (2) Convolution nets followed by fully connected layers
    (3) Consecutive nets (Fully convolutional nets)
    (4) Dilated Conv nets with dual pathway

'''

__author__ = 'cagdas'

from keras.models import Input, Model
from keras.layers import Conv2D, Conv3D, Dense, Concatenate, MaxPooling2D, Conv2DTranspose
from keras.layers import UpSampling2D, Dropout, BatchNormalization, PReLU
import numpy as np


def conv_block(m, dim, acti, bn, res, do=0.5):
    n = Conv2D(dim, 3, padding='same')(m)
    n = PReLU()(n)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, padding='same')(n)
    n = PReLU()(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else MaxPooling2D()(n)   
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, padding='same')(m)
            m = PReLU()(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=1, padding='same')(m)
            m = PReLU()(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m

def UNet(img_shape, out_ch=1, start_ch=32, depth=3, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, pooling=True, upconv=True, residual=True):

    '''
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (https://arxiv.org/abs/1505.04597)
    ---
    img_shape: (height, width, channels)
    out_ch: number of output channels
    start_ch: number of channels of the first conv
    depth: zero indexed depth of the U-structure
    inc_rate: rate at which the conv channels will increase
    activation: activation function after convolutions
    dropout: amount of dropout in the contracting part
    batchnorm: adds Batch Normalization if true
    maxpool: use strided conv instead of maxpooling if false
    upconv: use transposed conv instead of upsamping + conv if false
    residual: add residual connections around each conv block if true
    '''
    print('Using 2D U-net')
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, pooling, upconv, residual)
    o = Conv2D(out_ch, 1)(o)
    return Model(inputs=i, outputs=o)


def conv_fully_con_net(img_shape, start_ch=32, out_ch=2, kernel_dim=3, depth=4, batchnorm = False, act='relu'):
    print('Using convolutional + fully connected layers...')
    i = Input(shape=img_shape)
    cn = 1
    m  = Conv3D(cn*start_ch, kernel_dim, activation=act, padding='same')(i)
    m = BatchNormalization()(m) if batchnorm else m
    for k in range(depth-1):
         m  = Conv3D(cn*start_ch, kernel_dim, activation=act, dilation_rate = (k+1)**2, padding='same')(m)
         m = BatchNormalization()(m) if batchnorm else m
         
    n = Dense(256)(m)
    o = Dense(out_ch)(n)
    return Model(inputs=i, outputs=o)
    
    
def consecutive_net(img_shape, start_ch=64, out_ch=1, kernel_dim=3, depth=4, batchnorm = False):
    print('Using fully convolutional net..')
    i = Input(shape=img_shape)
    
    if np.size(img_shape) == 4:
        conv_2D = False
    else:
        conv_2D = True
    o = consecutive_blocks(i, start_ch, kernel_dim, depth, batchnorm, conv_2D)
    
    if conv_2D:
        o = Conv2D(out_ch, kernel_dim, padding='same')(o)
    else:
        o = Conv3D(out_ch, kernel_dim, padding='same')(o)
        
    
    return Model(inputs=i, outputs=o)


def consecutive_blocks(inp, channel_no, kernel_dim, depth, bn, conv_2D):

    if conv_2D:
        if depth > 0:
            m = Conv2D(channel_no, kernel_dim, padding='same')(inp)
            m = PReLU()(m)
            for i in range(depth):
                m = Conv2D(channel_no, kernel_dim, padding='same')(m)
                m = BatchNormalization(axis=-1)(m) if bn else m  ## Apply BN before activation
                m = PReLU()(m)
        else:
            m = Conv2D(channel_no, kernel_dim, padding='same')(inp)
            m = PReLU()(m)
    else:
        if depth > 0:
            m = Conv3D(channel_no, kernel_dim, padding='same')(inp)
            m = PReLU()(m)
            for i in range(depth):
                m = Conv3D(channel_no, kernel_dim, padding='same')(m)
                m = BatchNormalization(axis=-1)(m) if bn else m  ## Apply BN before activation
                m = PReLU()(m)
        else:
             m = Conv3D(channel_no, kernel_dim, padding='same')(inp)
             m = PReLU()(m)

    return m



def dilated_conv_net(img_shape, channel_no = 64, out_ch = 1, kernel_dim=3, depth=4, bn = False, act ='relu'):
    print('Using dilated convolutions..')

    inp = Input(shape=img_shape)
    m = Conv2D(channel_no, kernel_dim, activation=act, padding='same')(inp)
    
    k=m
    l=m
    
    for i in range(depth):
        k = Conv2D(channel_no, kernel_dim, activation=act, padding = 'same')(k)
        k = BatchNormalization()(k) if bn else k 
      
    
    for i in range(depth):
        l = Conv2D(channel_no, kernel_dim, activation=act, dilation_rate= 2**(i+1), padding = 'same')(l)
        l = BatchNormalization()(l) if bn else l
  
    
    c = Concatenate()([k,l])
    
    c = Conv2D(2*channel_no, kernel_dim, activation=act, padding='same')(c)
    
    o = Conv2D(out_ch, kernel_dim, padding='same')(c)
    
    return Model(inputs=inp, outputs=o)




