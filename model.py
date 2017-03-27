import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from random import randint
from math import tan , atan, radians, degrees
from PIL import Image

# Plot Histogram
# This function gives an insight of the data distribution based on the angle counts.

def plot_histogram(augmented_measurements, histogram_name):
    plt.figure(figsize=(7,4))
    plt.hist(augmented_measurements, 50, range=None)
    print(len(augmented_measurements))
    plt.savefig(histogram_name)
    print('Image '+histogram_name+' Saved')


# Read CSV File
# This function read the CSV file generated in the training phase
def leeCSV(ruta):
    lines = []
    with open(ruta) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


# Read Center, Left, Right image names and angle, Read adjust factor.
# This function read each csv row, and append the left and right image names, also calculate the angle of those images
# applying a correction factor.
# The correction factor takes as parameters the original angle for the center image ant the speed

def correction_factor(x, sp):
    center_camera_angle = (x*25)
    speed = sp
    center_camera_angle_rad = radians(center_camera_angle)
    left_camera_distance = 0.805  # in meters
    right_camera_distance = 0.826  # in meters
    forward_distance = ((speed * 1.60934) * 1000) / 3600  # speed in MPH converted to meters per second
    oposite_center = tan(center_camera_angle_rad) * forward_distance
    oposite_right = oposite_center - right_camera_distance
    oposite_left = oposite_center + left_camera_distance
    right_angle_grad = degrees(atan(oposite_right / forward_distance))
    left_angle_grad = degrees(atan(oposite_left / forward_distance))
    return right_angle_grad/25, left_angle_grad/25


def read_name_angle(lines):
    angles = []
    image_name = []
    for row in lines:
        speed = float(row[6])
        x = float(row[3])
        r_cf, l_cf = correction_factor(x, speed)
        image_name.append(row[0].split('/')[-1])
        angles.append(x)
        image_name.append(row[1].split('/')[-1])
        angles.append(l_cf)
        image_name.append(row[2].split('/')[-1])
        angles.append(r_cf)

    return (image_name, angles)

#Augmented images
# This function increase the counts of the hi low angle images.

def augment_images(image_name, angles_val):
    images = []
    angles = []

    for ima, ang in zip(image_name, angles_val):
        images.append(ima)
        angles.append(ang)

    for k in range(2):
        if (ang < -0.20 or ang > 0.20):
            images.append(ima)
            angles.append(ang)

    for l in range(20):
        if (ang < -0.35 or ang > 0.35):
            images.append(ima)
            angles.append(ang)

    return (images, angles)

def image_array(image_name, angle_original):
    current_path = '/home/salvador/aaSDCNDJ/test173/IMGR2/'+image_name
    ai = cv2.imread(current_path)
    ai = cv2.cvtColor(ai,cv2.COLOR_RGB2BGR)
    #ai = ai.resize((64, 64), Image.ANTIALIAS)
    r = randint(0,1)
    if r == 0:
        flipped_image = cv2.flip(ai, 1)
        im_to_return = flipped_image
        angle_to_return = float(angle_original) * -1.0
    else:
        im_to_return = ai
        angle_to_return = angle_original
    return (im_to_return, angle_to_return)


def generator(X_train,y_train, n ):
    while True:
        Xn = []
        yn = []
        for i in range(n):
            idx = randint(0, len(X_train)-1)
            Xi, yi = image_array(X_train[idx], y_train[idx])
            Xn.append(Xi)
            yn.append(yi)
        yield (np.asarray(Xn), np.asarray(yn))


# Neural network
# Call the needed keras methods.
# Accept name, epochs and learning rate as parameters

def keras_model(): #X_train, y_train, model_save_name, epochs=5, learning_rate=0.001):

    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
    from keras.layers.convolutional import Convolution2D
    from keras.layers.core  import Activation
    from keras.layers.advanced_activations import ELU


    model = Sequential()

    # Nvidia model architecture.
    model.add(Lambda(lambda x: x/255 -0.5, input_shape=(64,64,3)))
    model.add(Cropping2D(cropping=((10,0),(0,0))))
    model.add(Convolution2D(24,5,5, activation='relu', subsample=(2,2), border_mode='same'))
    model.add(Convolution2D(36,5,5,activation='relu', subsample=(2,2), border_mode='same'))
    model.add(Convolution2D(48,5,5,activation='relu',subsample=(2,2), border_mode='same'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(ELU())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))



    '''

    model.add(Convolution2D(12,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(24,3,3))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    '''
    return  model



# Pipeliene Function
# Parameters are defined here, and call the functions.
def test_run():
    # Parameters input area
    correction_angle = 0.11
    n = 15
    images_path='/home/salvador/aaSDCNDJ/test173/IMGR2/'
    csv_path = '/home/salvador/aaSDCNDJ/test173/driving_log.csv'
    model_save_name = 'nvidia3.h5'
    histogram_name = "Histogram_1.jpg"
    epochs_number = 5
    learning_rate = 0.0001
    # Pipeline.
    lines = leeCSV(csv_path)
    image_name, angles = read_name_angle(lines)
    X_train , y_train = augment_images(image_name, angles)

    #plot_histogram(augmented_measurements, histogram_name)



    adam = optimizers.Adam(lr=learning_rate, epsilon=1e-08)
    model = keras_model()
    model.compile(loss='mse', optimizer=adam)

    model.fit_generator(generator(X_train,y_train, 64), samples_per_epoch= 25600, nb_epoch = epochs_number, validation_data=generator(image_name, angles, 64), nb_val_samples=1600)

    model.save(model_save_name)


    #keras_model(X_train, y_train, model_save_name, epochs_number, learning_rate)


if __name__ == "__main__":
    test_run()
