import tensorflow as tf
import os

def model_placement(model, num_gpus):
    
    if num_gpus > 1:
        from tensorflow.python.keras.utils import multi_gpu_model
        
        with tf.device('/cpu:0'):
            p_model = model
        parallel_model = multi_gpu_model(p_model, gpus=num_gpus)
    
        return parallel_model
    
    elif num_gpus == 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        p_model = model
        return p_model
    
    else:
        with tf.device('/gpu:0'):
            p_model = model
        return p_model