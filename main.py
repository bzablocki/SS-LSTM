import matplotlib
matplotlib.use('Agg')
import cv2
import os
from load_data import image_tensor, all_image_tensor
from SSLSTM_model import EnsembleModel
from utils import preprocess
import data_process as dp
from keras.optimizers import Adam, RMSprop, SGD
#from matplotlib import pyplot as plt
from utils import circle_group_model_input, log_group_model_input, group_model_input
from utils import preprocess, get_traj_like, get_obs_pred_like, person_model_input, model_expected_ouput
import numpy as np
from load_data import load_test_data, print_test_data

#initialize model
model = EnsembleModel()

#Pre-trained model weighs path
pretrained_model_weights = 'sslstm_weights_checkpoint.h5'

#Model parameters
opt = RMSprop(lr=0.003)
batch_size = 20

#Load pre-trained weights
model.load_weights(pretrained_model_weights)
model.compile(optimizer=opt, loss='mse')

#Load data
test_input, expected_ouput, obs = load_test_data(dataset_index = 1, map_index = 3)

print('Predicting...')
predicted_output = model.predict(test_input, batch_size=batch_size, verbose=1)
print('Predicting Done!')

print_test_data(predicted_output, expected_ouput, obs)
