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

samples = []
data_location = '../udacity_data/' #'/opt/carnd_p3/data/'

for datafile in ['driving_log.csv', 'driving_log_shiv.csv']:
    with open(os.path.join(data_location, datafile)) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
random.seed(1234)
        
def generator(samples, batch_size=32):
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
                    check_dirs = [os.path.join(data_location, 'IMG' ,filename), 
                                  os.path.join(data_location, 'IMG_shiv' ,filename)]
                    current_path = None
                    for check_dir in check_dirs:
                        if os.path.exists(check_dir):
                            current_path =  check_dir

                    
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    if 'left' in filename:
                        measurement += 0.2
                    elif 'right' in filename:
                        measurement -= 0.2
                    measurements.append(measurement)
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    measurements.append(-measurement)


            X_train = np.array(images)
            y_train = np.array(measurements)
                
            yield sklearn.utils.shuffle(X_train, y_train)


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
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch)
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


    