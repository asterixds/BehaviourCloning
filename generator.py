import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random
import matplotlib.image as mpimg
from utils import pipeline

ANGLE_SHIFT = 0.3

"""get the raw image and steering angle data from the logs. The lef, right and center images are randomly
picked and ashift factor applied to account for the offsets of the left and right camera angles"""
def get_camera_data(batch):
    camera_data =[]
    for line in batch:
        if(len(line)>0):
            center_angle = float(line[3])
            rnd = np.random.randint(0, 3)
            path = line[rnd].split('/')[-1]
            filename = './data/IMG/'+path
            if (rnd ==1):
                camera_data.append((filename,center_angle + ANGLE_SHIFT))
            elif (rnd ==2):
                camera_data.append((filename,center_angle - ANGLE_SHIFT))
            else:
                camera_data.append((filename,center_angle ))
    return camera_data

"""Generate on-demand batches of images and label data. The generator applies the arandom ugmentation pipeline
which allows one to create almost infitite data sets from the raw data"""
def generator(records, batch_size=32, augment=False):
    while True:
        num_records = len(records)
        filter_indices = np.random.randint(0, num_records, batch_size)
        batch = np.array(records)[filter_indices]
        camera_data = get_camera_data(batch)
        images =[]
        angles = []
        for filename, angle in camera_data:
            img = mpimg.imread(filename)
            #apply the augmentation trasforms when training
            if augment:
                img, angle = pipeline(img, float(angle))
            images.append(img)
            angles.append(float(angle))

        X = np.array(images)
        y = np.array(angles)

        # output a batch increment
        yield X, y
