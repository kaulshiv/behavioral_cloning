import csv
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import ceil
import random
import sklearn

DATA_LOCATION = '../udacity_data/'


def get_samples():
    """
    Get training and validation data
    driving_log.csv contains the udacity provided data
    driving_log_shiv.csv' contains data I generated by running the simulator 
    on track 1.

    Params:
    ----------
    n/a

    Returns
    ----------
    train_samples : list of strings
        filenames of training samples
    validation_samples : list of strings
        filenames of validation samples
    """
    samples = []

    for datafile in ['driving_log.csv', 'driving_log_shiv.csv']:
        with open(os.path.join(DATA_LOCATION, datafile)) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for line in reader:
                samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

      
def generator(samples, batch_size=32):
    """
    Coroutine to generate samples for keras' model fit function

    Note: Yields 6*batch_size samples at a time. For each "sample" in the batch 
    images are loaded from the right, left, and center cameras. For each of 
    these images their flipped counterparts are appended as well. Thus 6x the 
    data is generated from a single image.

    Params:
    ----------
    samples: list of strings
        filenames of all images from which to draw batches
    batch_size: int
        desired batch size, default 32

    Returns (yields)
    ----------
    XX: np.array of shape (batch_size*6, 160, 320, 3)
        collection of images
    yy: np.array of shape (batch_size*6)
        labels corresponding to images held in XX
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                for source_path in batch_sample[0:3]:
                    source_path = batch_sample[0]
                    filename = source_path.split('/')[-1]
                    check_dirs = [os.path.join(DATA_LOCATION, 'IMG' ,filename), 
                                  os.path.join(DATA_LOCATION, 'IMG_shiv' ,filename)]
                    current_path = None
                    for check_dir in check_dirs:
                        if os.path.exists(check_dir):
                            current_path =  check_dir

                    
                    image = cv2.imread(current_path)
                    measurement = float(batch_sample[3]) # steering angle measurement

                    # Use the left and right cameras to generate additional data
                    # Add a correction factor so the car steers away from the edges of the road 
                    if 'left' in filename:
                        measurement += 0.2
                    elif 'right' in filename:
                        measurement -= 0.2

                    images.append(image)
                    measurements.append(measurement)

                    # add a flipped version of the image for further data augmentation
                    # flip the steering angle as well
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    measurements.append(-measurement)


            XX = np.array(images)
            yy = np.array(measurements)
                
            yield sklearn.utils.shuffle(XX, yy)


def get_model():
    """
    Implement model similar to the one used by NVIDIA in their paper 
    "End to End Learning for Self-Driving Cars".

    Params: 
    ----------
    n/a

    Returns
    ----------
    train_samples : keras.Model()
        final model architecture
    """

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24,5,strides=(2,2),activation="relu"))
    model.add(Conv2D(36,5,strides=(2,2),activation="relu"))
    model.add(Conv2D(48,5,strides=(2,2),activation="relu"))
    model.add(Conv2D(64,3,activation="relu"))
    model.add(Conv2D(64,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def train_and_save_model(model, train_samples, validation_samples):
    """
    Train and save model. Save picture of loss curves.

    Params: 
    ----------
    model: keras.Model()
        model to train!
    train_samples : list of strings
        filenames of training samples
    validation_samples : list of strings
        filenames of validation samples

    Returns
    ----------
    n/a
    """
    batch_size = 32
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    history_object = model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size),
                                        validation_data=validation_generator,
                                        validation_steps=ceil(len(validation_samples)/batch_size),
                                        epochs=5, verbose=1)

    model.save('model_data_augmentation.h5')

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('loss.png')


if __name__ == '__main__':
    train_samples, validation_samples = get_samples()
    model = get_model()
    random.seed(1234)
    train_and_save_model(model, train_samples, validation_samples)
