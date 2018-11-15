# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam, RMSprop, SGD
import time
import cv2
import os
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from SSLSTM_model import EnsembleModel
from load_data import load_data, DataGenerator
from keras import backend as K
from utils import EarlyStopping

def loss(y_true, y_pred):
    print('##############################')
    print(y_pred[2])
    return K.mean(K.square(y_pred - y_true), axis=-1)*(K.square(y_pred[:][-1] - y_true[:][-1]))

if __name__ == '__main__':
    #Allocate memory dinamically
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    #initialize model
    model = EnsembleModel()

    #parallel_model = multi_gpu_model(model, gpus = 4)

    #model parameters
    #opt = RMSprop(lr=0.003)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    batch_size = 20
    epochs = 1000

    #Load Data
    person_input, expected_output, group_input, scene_input, test_input, test_output = load_data()#[0:4]

    #Set data generator
    train_generator = DataGenerator(data = [scene_input, group_input, person_input], labels = expected_output, batch_size = batch_size, shuffle = False)
    test_generator = DataGenerator(data = test_input, labels = test_output, batch_size = batch_size, shuffle = False)

    #Compile model
    model.compile(loss='mse', optimizer=opt)

    #Setting TensorBoard
    tbCallback = TensorBoard(log_dir='graph/', histogram_freq=0, write_graph=False, write_images=False)

    #Settig CheckpointCallback
    mcpCallback = ModelCheckpoint('sslstm_weights_checkpoint.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    #Settig EarlyStoppingtCallback
    esCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, start_epoch = 300)
    
    #training
    
    model.fit_generator(train_generator,
                        validation_data=test_generator,
                        max_queue_size=1, 
                        workers=10, 
                        use_multiprocessing=False,
                        epochs=epochs, 
                        verbose=1,
                        callbacks=[tbCallback, mcpCallback, esCallback])
    '''
    model.fit([scene_input, group_input, person_input], expected_output,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            #validation_data=[test_input, test_output],
                            shuffle=False)
    '''
    #saving model after training
    model.save_weights('sslstm_weights.h5')