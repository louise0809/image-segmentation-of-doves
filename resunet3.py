# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:35:47 2021

@author: Admin
"""
from tensorflow.keras import Input, optimizers, datasets, Sequential,Model,regularizers
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dropout,BatchNormalization,UpSampling2D,Conv2DTranspose,Add,Concatenate
from tensorflow import concat
import tensorflow as tf

def bn_act(x, act=True):
    'batch normalization layer with an optinal activation layer'
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation('relu')(x)
    return x
def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    'convolutional layer which always uses the batch normalization layer'
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv
def stem(x, filters, kernel_size=3, padding='same', strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([conv, shortcut])
    return output
def residual_block(x, filters, k_size=3, padding='same', strides=1):
    res = conv_block(x, filters, k_size, padding, strides)
    res = conv_block(res, filters, k_size, padding, 1)
    shortcut = Conv2D(filters, k_size, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output
def upsample_concat_block(x, xskip):
    u = UpSampling2D((2,2))(x)
    c = Concatenate()([u, xskip])
    return c
def ResUNet(img_size=512):
    img_size=512
    f = [16, 32, 64, 128, 256, 512, 1024, 2048] * 32
    inputs = Input((img_size, img_size, 3))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    e6 = residual_block(e5, f[5], strides=2)
    e7 = residual_block(e6, f[6], strides=2)
    
    ## Bridge
    b0 = conv_block(e7, f[6], strides=1)
    b1 = conv_block(b0, f[6], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e6)
    d1 = residual_block(u1, f[6])
    
    u2 = upsample_concat_block(d1, e5)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e4)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e3)
    d4 = residual_block(u4, f[1])
    
    u5 = upsample_concat_block(d4, e2)
    d5 = residual_block(u5, f[1])
    
    u6 = upsample_concat_block(d5, e1)
    d6 = residual_block(u6, f[1])
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d6)
    model = tf.keras.models.Model(inputs, outputs)
    return model
# def dsc(y_true, y_pred):
#     smooth = 1.
#     y_true_f = Flatten()(y_true)
#     y_pred_f = Flatten()(y_pred)
#     intersection = reduce_sum(y_true_f * y_pred_f)
#     score = (2. * intersection + smooth) / (reduce_sum(y_true_f) + reduce_sum(y_pred_f) + smooth)
#     return score

# def dice_loss(y_true, y_pred):
#     loss = 1 - dsc(y_true, y_pred)
#     return loss

# def bce_dice_loss(y_true, y_pred):
#     loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
#     return loss
# model = ResUNet(img_size)
# adam = tf.keras.optimizers.Adam(lr = 0.01, epsilon = 0.1)
# model.compile(optimizer=adam, loss=bce_dice_loss, metrics=[dsc])