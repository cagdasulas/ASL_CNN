
# Import this if a GPU /CUDA is available in your machine.
#---------------------------------------------------------
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#--------------------------------------------------------

import matplotlib.pyplot as plt
from cnn_nets import consecutive_net
import numpy as np
from keras.optimizers import Adam
from asl_loss import asl_loss
from asl_loss import cbf_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import keras.backend as K
import time as tt



sess = tf.Session()

def model_train(x_train, y_train, x_val, y_val, param, net_depth, filter_dim, lambda_val, learning_rate):
    nb,nx,ny,nc = np.shape(x_train)


    model = consecutive_net((nx,ny,nc), start_ch = 48, out_ch = 1, kernel_dim = filter_dim, depth = net_depth, batchnorm = False)
    dec = 1e-3
    opt = Adam(lr=learning_rate, decay=dec)
    print('Learning rate = %s' %learning_rate)
    

    alpha = tf.constant(param['param'].alpha, dtype=np.float32)
    lambda_blood = tf.constant(param['param'].lambda_blood, dtype=np.float32)
    T1blood = tf.constant(param['param'].T1blood, dtype=np.float32)
    PLD = tf.constant(param['param'].PLDs, dtype=np.float32)
    tao = tf.constant(param['param'].labelduration, dtype=np.float32)
    scalar_constant = tf.constant(param['param'].scalar_constant, dtype=np.float32)
    reg_val = tf.constant(lambda_val, dtype=np.float32)

    print('Lambda = %s' %lambda_val)

    dims = np.shape(y_train)
    batch_size = 500

    y_train  = np.reshape(y_train, (y_train.shape[0], ) + (np.prod(y_train.shape[1:3]),)+y_train.shape[3:])
    y_val =  np.reshape(y_val, (y_val.shape[0],) + (np.prod(y_val.shape[1:3]),)+y_val.shape[3:])

     
    
    model.compile(optimizer=opt, loss=asl_loss(reg_val, alpha, lambda_blood, T1blood, PLD, tao, scalar_constant, dims, batch_size),
                  metrics=[psnr_metric(dims, batch_size), 
                           rmse_metric(reg_val, alpha, lambda_blood, T1blood, PLD, tao, scalar_constant, dims, batch_size)])


    cb = [EarlyStopping(monitor='val_loss', patience=20, mode='min'), ModelCheckpoint('weights.best.hdf5', save_best_only=True, monitor='val_loss', mode='min')]


    print('Network training is starting..')
    start_time = tt.time()
    print('Network depth is %d ..' %net_depth)
    print(np.shape(x_train))

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=300, validation_data=(x_val, y_val),
                        shuffle=True, callbacks = cb)

    end_time = tt.time()

    elapsed_time = end_time-start_time


    print('Network fitting finished..')
    print('Elapsed time is %.2f seconds..' %elapsed_time)

    return model, history



def model_predict(x_test, y_test, model):
    
    y_pred = model.predict(x_test, batch_size=1000)

    return y_pred



def rmse_metric(reg_val, alpha, lambda_blood, T1blood, PLD, tao, scalar_constant, dims, batch_size):
    def cbf_rmse(y_true, y_pred):
        y_true = tf.reshape(y_true, (batch_size,) + dims[1:])

        noisy = y_true[:,0,:,:,:]
        M0 = y_true[:,2,:,:,:]
        cbf_target = y_true[:,3,:,:,:]
        mask = y_true[:,4,:,:,:]
        
    
        im_pred = noisy - y_pred
        
        cbf_est = cbf_model(im_pred, M0, alpha, lambda_blood, T1blood, PLD, tao, scalar_constant)
        
        dif =  tf.abs(tf.reshape(cbf_target*mask, [-1]) - tf.reshape(cbf_est*mask, [-1]))
        return tf.sqrt(tf.reduce_mean(tf.square(dif)))
    return cbf_rmse


def psnr_metric(dims, batch_size):
    def im_psnr(y_true, y_pred):
        y_true = tf.reshape(y_true, (batch_size,) + dims[1:])

        noisy = y_true[:,0,:,:,:]
        y_target = y_true[:,1,:,:,:]
        mask = y_true[:,4,:,:,:]

        
        y_pred = y_pred - 0.0001*y_pred
        
        y_grnd = (noisy - y_target)*mask
        y_est = (noisy - y_pred)*mask
        mse = tf.reduce_mean(tf.square(tf.abs(tf.reshape(y_grnd, [-1]) -
                                              tf.reshape(y_est, [-1]))))
        return 10*log10(1.0/mse)
    return im_psnr



def ssim_metric(dims, batch_size):
    def metric(y_true, y_pred):
        y_true = tf.reshape(y_true, (batch_size,) + dims[1:])
        
        noisy = y_true[:,0,:,:,:]
        y_target = y_true[:,1,:,:,:]
        y_grnd = noisy - y_target
        y_est = noisy - y_pred
        
        y_grnd = tf.reshape(y_grnd, (tf.shape(y_grnd)[0], -1))
        y_est = tf.reshape(y_est, (tf.shape(y_est)[0], -1))
        
        u_true = K.mean(y_grnd, axis=1)
        u_pred = K.mean(y_est, axis=1)
        var_true = K.var(y_grnd, axis=1)
        var_pred = K.var(y_est, axis=1)
        std_true = K.sqrt(var_true)
        std_pred = K.sqrt(var_pred)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
        denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
        ssim /= denom
        ssim = tf.where(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
        return tf.reduce_mean(ssim)
    return metric 


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def plot_stats(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(True)
    plt.savefig("asl-train-validation-loss.png")
    plt.show()

