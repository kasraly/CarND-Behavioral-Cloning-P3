import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle


steering = []
training_folder = 'training3'
with open(os.path.join(os.curdir, training_folder, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering.append(float(line[3]))
steering = np.array(steering)
steer_filt = np.zeros(np.shape(steering))
for i in range(0, len(steering)):
    steer_filt[i] = steering[max(0, i - 3):min(i + 4, len(steering))].mean()

#plt.plot(range(0, len(steering)), steering, range(0, len(steering)), steer_filt)
# plt.show()

samples = []
with open(os.path.join(os.curdir, training_folder, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for line in reader:
        line[3] = str(steer_filt[i])
        samples.append(line + ['0'] + ['0'])  # original image
        samples.append(line + ['0'] + ['1'])  # flipped image
        samples.append(line + ['-1'] + ['0'])  # left image
        samples.append(line + ['-1'] + ['1'])  # flipped left image
        samples.append(line + ['1'] + ['0'])  # right image
        samples.append(line + ['1'] + ['1'])  # flipped right image
        i = i + 1

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if (batch_sample[-2] == '0'):
                    filepath, filename = os.path.split(batch_sample[0])
                    angle = float(batch_sample[3])
                elif (batch_sample[-2] == '-1'):
                    filepath, filename = os.path.split(batch_sample[1])
                    angle = float(batch_sample[3]) + 0.15
                else:
                    filepath, filename = os.path.split(batch_sample[2])
                    angle = float(batch_sample[3]) - 0.15

                image_path = os.path.join(
                    os.curdir, training_folder, 'IMG', filename)
                center_image = cv2.imread(image_path)
                if (batch_sample[-1] == '0'):
                    images.append(center_image)
                    angles.append(angle)
                else:
                    images.append(np.fliplr(center_image))
                    angles.append(-angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

a = (next(train_generator))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Cropping2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers.noise import GaussianNoise
from keras.layers.noise import GaussianDropout


model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0)))
model.add(GaussianNoise(0.1))

model.add(Conv2D(24, (5, 5), padding="valid", strides=(2, 2),
                 activation="relu", kernel_initializer="glorot_normal"))  # 24x36x158
model.add(Conv2D(36, (5, 5), padding="valid", strides=(2, 2),
                 activation="relu", kernel_initializer="glorot_normal"))  # 36x16x77
model.add(Conv2D(48, (5, 5), padding="valid", strides=(2, 2),
                 activation="relu", kernel_initializer="glorot_normal"))  # 48x6x37
model.add(Conv2D(64, (4, 4), padding="valid", activation="relu",
                 kernel_initializer="glorot_normal"))  # 64x3x34
model.add(Conv2D(64, (3, 3), padding="valid", activation="relu",
                 kernel_initializer="glorot_normal"))  # 64x1x32

model.add(Flatten())
model.add(GaussianDropout(0.25))

model.add(Dense(100, activation="relu", kernel_initializer="glorot_normal"))
model.add(Dense(50, activation="relu", kernel_initializer="glorot_normal"))
model.add(Dense(10, activation="relu", kernel_initializer="glorot_normal"))

model.add(Dense(1, kernel_initializer="glorot_normal"))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                        steps_per_epoch=100,
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples) / 64,
                        epochs=25,
                        verbose=2)

model.save('model.h5')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
