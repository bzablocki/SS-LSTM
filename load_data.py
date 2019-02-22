import matplotlib
matplotlib.use('Agg')
from scipy.spatial import distance
import numpy as np
import time
import cv2
import os.path as path
import sys
from utils import circle_group_model_input, log_group_model_input, group_model_input
from utils import preprocess, get_traj_like, get_obs_pred_like, person_model_input, model_expected_ouput
import data_process as dp
from matplotlib import pyplot as plt
import keras

observed_frame_num = 8
predicting_frame_num = 12

dimensions_1 = [720, 576]
dimensions_2 = [640, 480]
img_width_1 = 720
img_height_1 = 576

neighborhood_size = 32
grid_size = 4
neighborhood_radius = 32
grid_radius = 4
# grid_radius_1 = 4
grid_angle = 45
circle_map_weights = [1, 1, 1, 1, 1, 1, 1, 1]

data_dir_1 = './data/eth/hotel/'
data_dir_2 = './data/eth/univ/'
data_dir_3 = './data/ucy/univ/'
data_dir_4 = './data/ucy/zara/zara01/'
data_dir_5 = './data/ucy/zara/zara02/'

frame_dir_1 = './data/eth/hotel/frames/'
frame_dir_2 = './data/eth/univ/frames/'
frame_dir_3 = './data/ucy/univ/frames/'
frame_dir_4 = './data/ucy/zara/zara01/frames/'
frame_dir_5 = './data/ucy/zara/zara02/frames/'
data_str_1 = 'ETHhotel-'
data_str_2 = 'ETHuniv-'
data_str_3 = 'UCYuniv-'
data_str_4 = 'zara01-'
data_str_5 = 'zara02-'


# img reading functions
def image_tensor(data_dir, data_str, frame_ID):
    img_dir = data_dir + data_str + str(frame_ID) + '.jpg'
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (720, 576))

    return img


def all_image_tensor(data_dir, data_str, obs, img_width, img_height):
    image = []

    for i in range(len(obs)):
        fig = image_tensor(data_dir, data_str, int(obs[i][-1][1]))
        image.append(fig)

    image = np.reshape(image, [len(obs), img_height, img_width, 3])

    return image


def load_data(leave_dataset_index = 1, map_index = 1):

    # data_dir_1
    raw_data_1, numPeds_1 = preprocess(data_dir_1)

    if(path.exists('./data/img_1.npy')):
        obs_1 = np.load('./data/obs_1.npy')
        pred_1 = np.load('./data/pred_1.npy')
        img_1 = np.load('./data/img_1.npy')
    else:
        check = dp.DataProcesser(data_dir_1, observed_frame_num, predicting_frame_num)
        obs_1 = check.obs
        pred_1 = check.pred
        img_1 = all_image_tensor(frame_dir_1, data_str_1,obs_1, img_width_1, img_height_1)
        
        np.save('./data/obs_1.npy', obs_1)
        np.save('./data/pred_1.npy', pred_1)
        #np.save('./data/img_1.npy', img_1)

    #img_1 = check.heatmap
    #img_1 = np.array([img_1,]*len(obs_1))

    person_input_1 = person_model_input(obs_1, observed_frame_num)
    expected_ouput_1 = model_expected_ouput(pred_1, predicting_frame_num)
    group_log_1 = log_group_model_input(obs_1, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                        grid_radius, grid_angle, circle_map_weights, raw_data_1)
    group_grid_1 = group_model_input(obs_1, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_1)
    group_circle_1 = circle_group_model_input(obs_1, observed_frame_num, neighborhood_size, dimensions_1,
                                            neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_1)

    # data_dir_2
    raw_data_2, numPeds_2 = preprocess(data_dir_2)

    if(path.exists('./data/img_2.npy')):
        obs_2 = np.load('./data/obs_2.npy')
        pred_2 = np.load('./data/pred_2.npy')
        img_2 = np.load('./data/img_2.npy')
    else:
        check = dp.DataProcesser(data_dir_2, observed_frame_num, predicting_frame_num)
        obs_2 = check.obs
        pred_2 = check.pred
        img_2 = all_image_tensor(frame_dir_2, data_str_2, obs_2, img_width_1, img_height_1)

        np.save('./data/obs_2.npy', obs_2)
        np.save('./data/pred_2.npy', pred_2)
        #np.save('./data/img_2.npy', img_2)

    #img_2 = check.heatmap
    #img_2 = np.array([img_2,]*len(obs_2))

    person_input_2 = person_model_input(obs_2, observed_frame_num)
    expected_ouput_2 = model_expected_ouput(pred_2, predicting_frame_num)
    group_log_2 = log_group_model_input(obs_2, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                        grid_radius, grid_angle, circle_map_weights, raw_data_2)
    group_grid_2 = group_model_input(obs_2, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_2)
    group_circle_2 = circle_group_model_input(obs_2, observed_frame_num, neighborhood_size, dimensions_1,
                                            neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_2)

    # data_dir_3
    raw_data_3, numPeds_3 = preprocess(data_dir_3)

    if(path.exists('./data/img_3.npy')):
        obs_3 = np.load('./data/obs_3.npy')
        pred_3 = np.load('./data/pred_3.npy')
        img_3 = np.load('./data/img_3.npy')
    else:
        check = dp.DataProcesser(data_dir_3, observed_frame_num, predicting_frame_num)
        obs_3 = check.obs
        pred_3 = check.pred
        img_3 = all_image_tensor(frame_dir_3, data_str_3, obs_3, img_width_1, img_height_1)

        np.save('./data/obs_3.npy', obs_3)
        np.save('./data/pred_3.npy', pred_3)
        #np.save('./data/img_3.npy', img_3)

    
    #img_3 = check.heatmap
    #img_3 = np.array([img_3,]*len(obs_3))

    person_input_3 = person_model_input(obs_3, observed_frame_num)
    expected_ouput_3 = model_expected_ouput(pred_3, predicting_frame_num)
    group_log_3 = log_group_model_input(obs_3, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                        grid_radius, grid_angle, circle_map_weights, raw_data_3)
    group_grid_3 = group_model_input(obs_3, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_3)
    group_circle_3 = circle_group_model_input(obs_3, observed_frame_num, neighborhood_size, dimensions_1,
                                            neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_3)
   
    # data_dir_4
    raw_data_4, numPeds_4 = preprocess(data_dir_4)

    if(path.exists('./data/img_4.npy')):
        obs_4 = np.load('./data/obs_4.npy')
        pred_4 = np.load('./data/pred_4.npy')
        img_4 = np.load('./data/img_4.npy')
    else:
        check = dp.DataProcesser(data_dir_4, observed_frame_num, predicting_frame_num)
        obs_4 = check.obs
        pred_4 = check.pred
        img_4 = all_image_tensor(frame_dir_4, data_str_4, obs_4, img_width_1, img_height_1)

        np.save('./data/obs_4.npy', obs_4)
        np.save('./data/pred_4.npy', pred_4)
        #np.save('./data/img_4.npy', img_4)
    
    #img_4 = check.heatmap
    #img_4 = np.array([img_4,]*len(obs_4))

    person_input_4 = person_model_input(obs_4, observed_frame_num)
    expected_ouput_4 = model_expected_ouput(pred_4, predicting_frame_num)
    group_log_4 = log_group_model_input(obs_4, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                        grid_radius, grid_angle, circle_map_weights, raw_data_4)
    group_grid_4 = group_model_input(obs_4, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_4)
    group_circle_4 = circle_group_model_input(obs_4, observed_frame_num, neighborhood_size, dimensions_1,
                                            neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_4)

    # data_dir_5
    raw_data_5, numPeds_5 = preprocess(data_dir_5)

    if(path.exists('./data/img_5.npy')):
        obs_5 = np.load('./data/obs_5.npy')
        pred_5 = np.load('./data/pred_5.npy')
        img_5 = np.load('./data/img_5.npy')
    else:
        check = dp.DataProcesser(data_dir_5, observed_frame_num, predicting_frame_num)
        obs_5 = check.obs
        pred_5 = check.pred
        img_5 = all_image_tensor(frame_dir_5, data_str_5, obs_5, img_width_1, img_height_1)

        np.save('./data/obs_5.npy', obs_5)
        np.save('./data/pred_5.npy', pred_5)
        #np.save('./data/img_5.npy', img_5)

    
    #img_5 = check.heatmap
    #img_5 = np.array([img_5,]*len(obs_5))

    person_input_5 = person_model_input(obs_5, observed_frame_num)
    expected_ouput_5 = model_expected_ouput(pred_5, predicting_frame_num)
    group_log_5 = log_group_model_input(obs_5, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                        grid_radius, grid_angle, circle_map_weights, raw_data_5)
    group_grid_5 = group_model_input(obs_5, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_5)
    group_circle_5 = circle_group_model_input(obs_5, observed_frame_num, neighborhood_size, dimensions_1,
                                            neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_5)


    if map_index == 1:
        group_input_1 = group_grid_1
        group_input_2 = group_grid_2
        group_input_3 = group_grid_3
        group_input_4 = group_grid_4
        group_input_5 = group_grid_5

    elif map_index == 2:
        group_input_1 = group_circle_1
        group_input_2 = group_circle_2
        group_input_3 = group_circle_3
        group_input_4 = group_circle_4
        group_input_5 = group_circle_5

    elif map_index == 3:
        group_input_1 = group_log_1
        group_input_2 = group_log_2
        group_input_3 = group_log_3
        group_input_4 = group_log_4
        group_input_5 = group_log_5

    if leave_dataset_index == 1:
        person_input = np.concatenate(
            (person_input_2, person_input_3, person_input_4, person_input_5))
        expected_ouput = np.concatenate(
            (expected_ouput_2, expected_ouput_3, expected_ouput_4, expected_ouput_5))
        group_input = np.concatenate((group_input_2, group_input_3, group_input_4, group_input_5))
        scene_input = np.concatenate((img_2, img_3, img_4, img_5))
        test_input = [img_1, group_input_1, person_input_1]
        test_output = expected_ouput_1

    elif leave_dataset_index == 2:
        person_input = np.concatenate(
            (person_input_1, person_input_3, person_input_4, person_input_5))
        expected_ouput = np.concatenate(
            (expected_ouput_1, expected_ouput_3, expected_ouput_4, expected_ouput_5))
        group_input = np.concatenate((group_input_1, group_input_3, group_input_4, group_input_5))
        scene_input = np.concatenate((img_1, img_3, img_4, img_5))
        test_input = [img_2, group_input_2, person_input_2]
        test_output = expected_ouput_2

    elif leave_dataset_index == 3:
        person_input = np.concatenate((person_input_1, person_input_2, person_input_4, person_input_5))
        expected_ouput = np.concatenate((expected_ouput_1, expected_ouput_2, expected_ouput_4, expected_ouput_5))
        group_input = np.concatenate((group_input_1, group_input_2, group_input_4, group_input_5))
        scene_input = np.concatenate((img_1, img_2, img_4, img_5))
        test_input = [img_3, group_input_3, person_input_3]
        test_output = expected_ouput_3

    elif leave_dataset_index == 4:
        person_input = np.concatenate((person_input_1, person_input_2, person_input_3, person_input_5))
        expected_ouput = np.concatenate((expected_ouput_1, expected_ouput_2, expected_ouput_3, expected_ouput_5))
        group_input = np.concatenate((group_input_1, group_input_2, group_input_3, group_input_5))
        scene_input = np.concatenate((img_1, img_2, img_3, img_5))
        test_input = [img_4, group_input_4, person_input_4]
        test_output = expected_ouput_4

    elif leave_dataset_index == 5:
        person_input = np.concatenate((person_input_1, person_input_2, person_input_3, person_input_4))
        expected_ouput = np.concatenate((expected_ouput_1, expected_ouput_2, expected_ouput_3, expected_ouput_4))
        group_input = np.concatenate((group_input_1, group_input_2, group_input_3, group_input_4))
        scene_input = np.concatenate((img_1, img_2, img_3, img_4))
        test_input = [img_5, group_input_5, person_input_5]
        test_output = expected_ouput_5

    #test_input = np.array(test_input)

    return person_input, expected_ouput, group_input, scene_input, test_input, test_output

def load_test_data(dataset_index = 1, map_index = 3):

    if dataset_index == 1:
        # data_dir_1
        raw_data_1, numPeds_1 = preprocess(data_dir_1)

        check = dp.DataProcesser(data_dir_1, observed_frame_num, predicting_frame_num)
        obs_1 = check.obs
        pred_1 = check.pred

        img_1 = all_image_tensor(frame_dir_1, data_str_1,obs_1, img_width_1, img_height_1)
        person_input_1 = person_model_input(obs_1, observed_frame_num)
        expected_ouput_1 = model_expected_ouput(pred_1, predicting_frame_num)
        group_log_1 = log_group_model_input(obs_1, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                            grid_radius, grid_angle, circle_map_weights, raw_data_1)
        group_grid_1 = group_model_input(obs_1, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_1)
        group_circle_1 = circle_group_model_input(obs_1, observed_frame_num, neighborhood_size, dimensions_1,
                                                neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_1)
        obs = obs_1

        if map_index == 1:
            group_input = group_grid_1
        elif map_index == 2:
            group_input = group_circle_1
        elif map_index == 3:
            group_input = group_log_1

        #img_1 = check.heatmap
        #img_1 = np.array([img_1,]*len(obs_1))

    elif dataset_index == 2:
        # data_dir_2
        raw_data_2, numPeds_2 = preprocess(data_dir_2)

        check = dp.DataProcesser(data_dir_2, observed_frame_num, predicting_frame_num)
        obs_2 = check.obs
        pred_2 = check.pred

        img_2 = all_image_tensor(frame_dir_2, data_str_2, obs_2, img_width_1, img_height_1)
        person_input_2 = person_model_input(obs_2, observed_frame_num)
        expected_ouput_2 = model_expected_ouput(pred_2, predicting_frame_num)
        group_log_2 = log_group_model_input(obs_2, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                            grid_radius, grid_angle, circle_map_weights, raw_data_2)
        group_grid_2 = group_model_input(obs_2, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_2)
        group_circle_2 = circle_group_model_input(obs_2, observed_frame_num, neighborhood_size, dimensions_1,
                                                neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_2)
        obs = obs_2

        if map_index == 1:
            group_input = group_grid_2
        elif map_index == 2:
            group_input = group_circle_2
        elif map_index == 3:
            group_input = group_log_2

        #img_2 = check.heatmap
        #img_2 = np.array([img_2,]*len(obs_2))

    elif dataset_index == 3:
        # data_dir_3
        raw_data_3, numPeds_3 = preprocess(data_dir_3)

        check = dp.DataProcesser(data_dir_3, observed_frame_num, predicting_frame_num)
        obs_3 = check.obs
        pred_3 = check.pred

        img_3 = all_image_tensor(frame_dir_3, data_str_3, obs_3, img_width_1, img_height_1)
        person_input_3 = person_model_input(obs_3, observed_frame_num)
        expected_ouput_3 = model_expected_ouput(pred_3, predicting_frame_num)
        group_log_3 = log_group_model_input(obs_3, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                            grid_radius, grid_angle, circle_map_weights, raw_data_3)
        group_grid_3 = group_model_input(obs_3, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_3)
        group_circle_3 = circle_group_model_input(obs_3, observed_frame_num, neighborhood_size, dimensions_1,
                                                neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_3)
        obs = obs_3

        if map_index == 1:
            group_input = group_grid_3
        elif map_index == 2:
            group_input = group_circle_3
        elif map_index == 3:
            group_input = group_log_3

        #img_3 = check.heatmap
        #img_3 = np.array([img_3,]*len(obs_3))
   
    elif dataset_index == 4:
        # data_dir_4
        raw_data_4, numPeds_4 = preprocess(data_dir_4)

        check = dp.DataProcesser(data_dir_4, observed_frame_num, predicting_frame_num)
        obs_4 = check.obs
        pred_4 = check.pred

        img_4 = all_image_tensor(frame_dir_4, data_str_4, obs_4, img_width_1, img_height_1)
        person_input_4 = person_model_input(obs_4, observed_frame_num)
        expected_ouput_4 = model_expected_ouput(pred_4, predicting_frame_num)
        group_log_4 = log_group_model_input(obs_4, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                            grid_radius, grid_angle, circle_map_weights, raw_data_4)
        group_grid_4 = group_model_input(obs_4, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_4)
        group_circle_4 = circle_group_model_input(obs_4, observed_frame_num, neighborhood_size, dimensions_1,
                                                neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_4)
        obs = obs_4

        if map_index == 1:
            group_input = group_grid_4
        elif map_index == 2:
            group_input = group_circle_4
        elif map_index == 3:
            group_input = group_log_4

        #img_4 = check.heatmap
        #img_4 = np.array([img_4,]*len(obs_4))

    elif dataset_index == 5:
        # data_dir_5
        raw_data_5, numPeds_5 = preprocess(data_dir_5)

        check = dp.DataProcesser(data_dir_5, observed_frame_num, predicting_frame_num)
        obs_5 = check.obs
        pred_5 = check.pred

        img_5 = all_image_tensor(frame_dir_5, data_str_5, obs_5, img_width_1, img_height_1)
        person_input_5 = person_model_input(obs_5, observed_frame_num)
        expected_ouput_5 = model_expected_ouput(pred_5, predicting_frame_num)
        group_log_5 = log_group_model_input(obs_5, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                            grid_radius, grid_angle, circle_map_weights, raw_data_5)
        group_grid_5 = group_model_input(obs_5, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_5)
        group_circle_5 = circle_group_model_input(obs_5, observed_frame_num, neighborhood_size, dimensions_1,
                                                neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_5)
        obs = obs_5

        if map_index == 1:
            group_input = group_grid_5
        elif map_index == 2:
            group_input = group_circle_5
        elif map_index == 3:
            group_input = group_log_5

        #img_5 = check.heatmap
        #img_5 = np.array([img_5,]*len(obs_5))

    #preparing test data   
    if dataset_index == 1:
        person_input = person_input_1
        expected_ouput = expected_ouput_1
        scene_input = img_1
    elif dataset_index == 2:
        person_input = person_input_2
        expected_ouput = expected_ouput_2
        scene_input = img_2
    elif dataset_index == 3:
        person_input = person_input_3
        expected_ouput = expected_ouput_3
        scene_input = img_3
    elif dataset_index == 4:
        person_input = person_input_4
        expected_ouput = expected_ouput_4
        scene_input = img_4
    elif dataset_index == 5:
        person_input = person_input_5
        expected_ouput = expected_ouput_5
        scene_input = img_5

    test_input = [scene_input, group_input, person_input]

    return test_input, expected_ouput, obs

def print_test_data(predicted_output, expected_output, obs):
    IDs = []

    for i in range(len(obs)-1):
        img_ID = int(obs[i][-1][1])
        img_ID_start =  img_ID - 70
        #img_ID_current = img_ID_start
        img_ID_current = img_ID_start + 80 

        # while img_ID_current < img_ID_start + 80:
        #     image = image_tensor(frame_dir_1,data_str_1,img_ID_current)
        #     #image = cv2.imread(image, cv2.IMREAD_COLOR)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #     plt.close()
        #     plt.figure(figsize=(20,12))
        #     plt.imshow(image)

        #     current_axis = plt.gca()
        #     plt.axis('off')
        #     filename = 'image_' + str(img_ID_current) + '.png' #'\\' on windows
        #     plt.savefig(path.join('./data', 'video', filename), bbox_inches='tight',transparent=True, pad_inches=0)
            
        #     img_ID_current += 1

        while img_ID_current < img_ID_start + 90:  #200
            

            filename = 'image_' + str(img_ID_current) + '.png' #'\\' on windows
            filepath = path.join('./data', 'video', filename)

            if img_ID_current in IDs:
                image = cv2.imread(filepath)
                image = cv2.resize(image, (720, 576))
            else:
                IDs.append(img_ID_current)
                image = image_tensor(frame_dir_1, data_str_1, img_ID_current)
                #image = cv2.imread(image, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.close()
            plt.figure(figsize=(20,12))
            plt.imshow(image)

            #current_axis = plt.gca()
            plt.gca()
            #plt.axis('off')
            # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            # plt.margins(0,0)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                        
            X, Y, X1, Y1 = [], [], [], []
            
            for j in range(12):
                X.append(predicted_output[i][j][0]*img_width_1)
                Y.append(predicted_output[i][j][1]*img_height_1)
                #X = list(filter(lambda x : x >= 0 and x < img_width_1, X))
                #Y = list(filter(lambda x : , Y))
                Xt, Yt = [], [] 
                for x,y in zip(X, Y):
                    if x >= 0 and x < img_width_1 and y >= 0 and y < img_height_1:
                        Xt.append(x)
                        Yt.append(y)

                X = Xt[:]
                Y = Yt[:]

                #X1.append(expected_output[i][j][0]*img_width_1)
                #Y1.append(expected_output[i][j][1]*img_height_1)

            #X.append(predicted_output[1][i][0]*img_width_1)
            #Y.append(predicted_output[1][i][1]*img_height_1)

            #X1.append(expected_output[i][11][0]*img_width_1)
            #Y1.append(expected_output[i][11][1]*img_height_1)

            color = 'red'

            plt.plot(X, Y, color=color, linestyle='-', linewidth=4, markersize=4, marker='P', markerfacecolor='k')
            #plt.plot(X1, Y1, color='k', linestyle='-', linewidth=4, markersize=4, marker='P', markerfacecolor=color)
            count = 0
            for x, y in zip(X, Y):
                plt.text(x, y, str(count), color="blue", fontsize=12)
                count += 1
            count = 0
            # for x, y in zip(X1, Y1):
            #     plt.text(x, y, str(count), color="blue", fontsize=12)
            #     count += 1
            filename = 'image_' + str(img_ID_current) + '.png' #'\\' on windows
            #plt.savefig(path.join('./data', 'video', filename), bbox_inches='tight',transparent=True, pad_inches=0)
            save(path.join('./data', 'video', filename))
            img_ID_current += 1

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, batch_size, shuffle=True):
        'Initialization'
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        #Initialization
        scene_input = [self.data[0][k] for k in indexes]
        group_input = [self.data[1][k] for k in indexes]
        person_input = [self.data[2][k] for k in indexes]
        output = [self.labels[k] for k in indexes]
        #out1 = []
        #out2 = []
        #for k in range(0, len(output)):
        #    out1.append(output[k][0:11])
        #   out2.append(output[k][11])

        #print(out2)
        #Convert data to nparray
        scene_input = np.array(scene_input)
        group_input = np.array(group_input)
        person_input = np.array(person_input)   
        output = np.array(output)   
        #out1 = np.array(out1)
        #out2 = np.array(out2)
        #out1 = out1.reshape(20, 11, 2)    
        #out2 = out2.reshape(20, 1, 2)
       
        return [scene_input, group_input, person_input], output

def save(filepath, fig=None):
    '''Save the current image with no whitespace
    Example filepath: "myfig.png" or r"C:\myfig.pdf" 
    '''
    #import matplotlib.pyplot as plt
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0,0,1,1,0,0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(filepath, pad_inches = 0, bbox_inches='tight')