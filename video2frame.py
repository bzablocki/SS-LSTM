import cv2
import os.path as path

#DATA_PATH = 'd:data'
DATA_PATH = './data'
dir = 'ucy/zara/zara01/frames/'
path_dir = path.join(DATA_PATH, dir, 'crowds_zara01.avi')
print(path_dir)
#video to extract
vidcap = cv2.VideoCapture(path_dir)
print(vidcap)
# Find the number of frames
video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
success, image = vidcap.read()
print(video_length)
count = 0
while count < (video_length-1):
	#if the frame was extracted with success, save it
	if(success):
                filename = format(count, '06') + '.jpg'
                # save frame as JPEG file
                cv2.imwrite(path.join(DATA_PATH, dir, 'zara01-' + str(count) + '.jpg'), image)
                success, image = vidcap.read()
                count += 1

