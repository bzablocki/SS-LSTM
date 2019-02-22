import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, LSTM, GRU
from keras.layers import Add, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.layers import concatenate, add
from keras.layers.core import Permute
from keras.layers.core import RepeatVector
from keras.layers.wrappers import TimeDistributed

observed_frame_num = 8
predicting_frame_num = 12

hidden_size = 128
tsteps = observed_frame_num

img_width = 720
img_height = 576
img_channels = 3

batch_size = 20

neighborhood_size = 32
grid_size = 4
neighborhood_radius = 32
grid_radius = 4
grid_angle = 45
circle_map_weights = [1, 1, 1, 1, 1, 1, 1, 1]

# CNN model for scene
def Scene_CNN():
    img_shape = (img_height, img_width, img_channels)
    
    model = Sequential()
    #model.add(Input(shape=img_shape))
    #model.add(BatchNormalization(momentum=0.8))
    #model.add(Conv2D(96, kernel_size=11, strides=4, padding="same"))
    model.add(Conv2D(96, kernel_size=11, strides=4, input_shape=img_shape, padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=5, strides=1, padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))

    return model

def SceneModel():
    scene_model = Scene_CNN()
    scene_model.add(RepeatVector(tsteps))
    scene_model.add(GRU(hidden_size,
                        input_shape=(tsteps, 512),
                        batch_size=batch_size,
                        return_sequences=False,
                        stateful=False,
                        dropout=0.2))
    return scene_model

def SocialModel():
    group_model = Sequential()
    group_model.add(Dense(hidden_size, activation='relu', input_shape=(tsteps, 64)))
    group_model.add(GRU(hidden_size,
                        input_shape=(tsteps, int(neighborhood_radius / grid_radius) * int(360 / grid_angle)),
                        batch_size=batch_size,
                        return_sequences=False,
                        stateful=False,
                        dropout=0.2))
   
    return group_model

def PersonModel():
    person_model = Sequential()
    person_model.add(Dense(hidden_size, activation='relu', input_shape=(tsteps, 2)))
    person_model.add(GRU(hidden_size,
                         input_shape=(tsteps, 2),
                         batch_size=batch_size,
                         return_sequences=False,
                         stateful=False,
                         dropout=0.2))

    return person_model

def EnsembleModel():
    scene_model = SceneModel()
    social_model = SocialModel()
    person_model = PersonModel()

    scene_input = Input(shape=(img_height, img_width, img_channels))
    social_input = Input(shape= (tsteps, 64))
    person_input = Input(shape= (tsteps, 2))

    scene_model = scene_model(scene_input)
    social_model = social_model(social_input)
    person_model = person_model(person_input)

    input_braches = add([scene_model, social_model, person_model])

    x = RepeatVector(predicting_frame_num)(input_braches)
    x = GRU(128,
                  input_shape=(predicting_frame_num, 2),
                  batch_size=batch_size,
                  return_sequences=True,
                  stateful=False,
                  dropout=0.2)(x)
    x = TimeDistributed(Dense(2))(x)
    
    output = Activation('linear')(x)

    #r = Flatten()(x)
    #output2 = Dense(2, activation = 'linear')(r)

    model = Model(inputs = [scene_input, social_input, person_input], outputs = output)

    return model