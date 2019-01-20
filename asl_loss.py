
__author__ = 'Cagdas Ulas'


import tensorflow as tf


def asl_loss(reg_val, alpha, lambda_blood, T1blood, PLD, tao, scalar_constant, dims, batch_size):
    def calculate_loss(y_true, y_pred):
        
        y_true = tf.reshape(y_true, (batch_size,) + dims[1:])
        
        noisy = y_true[:,0,:,:,:]
        
        res_target = y_true[:,1,:,:,:]
        
        M0 = y_true[:,2,:,:,:]
        cbf_target = y_true[:,3,:,:,:]
        
        mask = y_true[:,4,:,:,:] # Brain mask 
        
        im_pred = noisy - y_pred

        cbf_est = cbf_model(im_pred, M0, alpha, lambda_blood, T1blood, PLD, tao, scalar_constant)
       
        # The loss value is calculated only within brain region
        loss = reg_val*rmse_loss_l2(res_target*mask, y_pred*mask) + (1-reg_val)*rmse_loss_l2(cbf_target*mask, cbf_est*mask)
        
        return loss   
    return calculate_loss



def cbf_model(y_pred, M0, alpha, lambda_blood, T1blood, PLD, tao, scalar_constant):

    # Convert from perfusion-weighted image to cerebral blood flow (CBF)
    cbf_est = (scalar_constant*lambda_blood*y_pred*tf.exp(PLD/T1blood)) / (2*alpha*T1blood*M0*(1-tf.exp(-(tao/T1blood))))

    return cbf_est



def rmse_loss_l2(targets, outputs):
    return tf.square(tf.abs(targets - outputs))   


def rmse_loss_l1(targets, outputs):
    return tf.abs(targets - outputs)
