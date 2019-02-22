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
import sys
from utils import calculate_ADE, calculate_FDE
import tensorflow as tf
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

#Allocate memory dinamically
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

#initialize model
model = EnsembleModel()

#Pre-trained model weighs path
pretrained_model_weights = 'sslstm_weights_checkpoint.h5'

#Model parameters
#opt = RMSprop(lr=0.003)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
batch_size = 20

#Parameters
show_num = 1
map_index = 2
dataset_index = 3

#Load pre-trained weights
model.load_weights(pretrained_model_weights)
model.compile(optimizer=opt, loss='mse')

#Load data
test_input, expected_output, obs = load_test_data(dataset_index = dataset_index, map_index = map_index)
print(test_input[2].shape)

# print(expected_ouput[1])

#sys.exit()
print('Predicting...')
predicted_output = model.predict(test_input, batch_size=batch_size, verbose=1)
print('Predicting Done!')

print('Calculating Predicting Error...')
mean_FDE = calculate_FDE(expected_output, predicted_output, len(expected_output), show_num)
mean_ADE = calculate_ADE(expected_output, predicted_output, len(expected_output), 12, show_num)
all_FDE = calculate_FDE(expected_output, predicted_output, len(expected_output), len(expected_output))
all_ADE = calculate_ADE(expected_output, predicted_output, len(expected_output), 12, len(expected_output))
print('ssmap_' + str(map_index) + '_ETHUCY_' + str(dataset_index) + 'ADE:', mean_ADE)
print('ssmap_' + str(map_index) + '_ETHUCY_' + str(dataset_index) + 'FDE:', mean_FDE)
print('ssmap_' + str(map_index) + '_ETHUCY_' + str(dataset_index) + 'all ADE:', all_ADE)
print('ssmap_' + str(map_index) + '_ETHUCY_' + str(dataset_index) + 'all FDE:', all_FDE)

#print_test_data(predicted_output, expected_output, obs)
