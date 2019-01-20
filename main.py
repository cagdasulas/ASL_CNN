# This main python module process training of a CNN model for denoising synthetic ASL and evaulate the trained model on the test data.
# The implementation is based on Keras framework with Tensorflow backend. Please make sure that Keras and Tensorflow are installed 
# to your machine in order to run this implementation successfully.



# Import this if a GPU /CUDA is available in your machine.
#---------------------------------------------------------
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#--------------------------------------------------------

import scipy.io as sc
import numpy as np
import asl_model_process as mp
import time as tt



def load_data(filename):
    return sc.loadmat(filename,squeeze_me=True, struct_as_record=False)

def split_data(sub_data, batch_per_sub, test_sub_ind, val_rate = 0.15):
    
    data = sub_data['net_data'].xdata
    label = sub_data['net_data'].ylabel
    M0 = sub_data['net_data'].M0
    cbf = sub_data['net_data'].CBF
    mask = sub_data['net_data'].mask
    
    test_sub_ind = test_sub_ind - 1
    test_ind = np.arange((test_sub_ind)*batch_per_sub, (test_sub_ind+1)*batch_per_sub)
    samples_no, px, py = np.shape(data)
    train_interval = np.setdiff1d(np.arange(samples_no), test_ind)

    rand_ind = np.random.permutation(train_interval)
    nval = int(np.round(val_rate*np.size(train_interval)))
    train_ind = rand_ind[:-nval]
    val_ind =  rand_ind[-nval:]
    
    
    nparam = 5
    
    x_train = np.zeros((np.size(train_ind[0:18000]), px, py, 1))
    x_val = np.zeros((np.size(val_ind[0:3000]), px, py, 1))
    x_test = np.zeros((np.size(test_ind), px, py, 1))
    
    x_train[:,:,:,0] = data[train_ind[0:18000],:,:]
    x_val[:,:,:,0] =  data[val_ind[0:3000],:,:]
    x_test[:,:,:,0] = data[test_ind,:,:]
    
    y_train = np.zeros((np.size(train_ind[0:18000]), nparam, px, py, 1))
    y_val = np.zeros((np.size(val_ind[0:3000]), nparam, px, py, 1))
    
    
    y_train[:,0,:,:,0] = data[train_ind[0:18000],:,:]
    y_train[:,1,:,:,0] = data[train_ind[0:18000],:,:] - label[train_ind[0:18000],:,:]
    y_train[:,2,:,:,0] = M0[train_ind[0:18000],:,:]
    y_train[:,3,:,:,0] = cbf[train_ind[0:18000],:,:]
    y_train[:,4,:,:,0] = mask[train_ind[0:18000],:,:]
    
    
    y_val[:,0,:,:,0] = data[val_ind[0:3000],:,:]
    y_val[:,1,:,:,0] = data[val_ind[0:3000],:,:,] - label[val_ind[0:3000],:,:]
    y_val[:,2,:,:,0] = M0[val_ind[0:3000],:,:]
    y_val[:,3,:,:,0] = cbf[val_ind[0:3000],:,:]
    y_val[:,4,:,:,0] = mask[val_ind[0:3000],:,:]
    
    y_test = data[test_ind] - label[test_ind,:,:]
    
    return x_train, y_train, x_val, y_val, x_test, y_test
    


def main():    
    data = load_data('synthetic_net_data.mat')
    params = load_data('synthetic_common_vars.mat')
   
    data_type = 'synthetic'
    
    lambda_val = 0.5
    sub_ind = 2

    batch_per_sub = params['param'].subject_batch_size
    # Split data into training, validation and test dataset
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(data, batch_per_sub, test_sub_ind = sub_ind)

    print('Training and test data were generated')

    print('Running a %s data of Subject %s' %(data_type, sub_ind))

    # Train the model with training data and validate on the validation dataset
    model, history = mp.model_train(x_train, y_train, x_val, y_val, params, net_depth=8, filter_dim=3, lambda_val=lambda_val, learning_rate=0.001)

    start_time = tt.time()
    
    # Prediction on the test dataset
    y_pred = mp.model_predict(x_test, y_test, model)

    end_time = tt.time()
    elapsed_time = end_time-start_time

    print('Prediction on test data finished..')
    print('Elapsed time is %.2f seconds..' %elapsed_time)

    # Save the results as .mat file for further analysis
    # The denoised image computed as x_test - y_pred, reference image as x_test - y_test.
    res={}

    res['y_test'] = y_test
    res['y_pred'] = y_pred
    res['x_test'] = x_test

    str =  'asl_result_%s_sub%s_lambda%s.mat' % (data_type, sub_ind, lambda_val)
        
    sc.savemat(str, {'res': res})


if __name__ == '__main__':
    # If GPU device available, comment out the following line..
    # with K.tf.device('/gpu:1'):  
    main()





