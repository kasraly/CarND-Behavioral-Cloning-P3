import csv
import cv2
import numpy as np
import os

images = []
measurements = []
training_dirs = ['Training1', 'Training3']
for training_dir in training_dirs:
    lines = []
    with open(os.path.join(os.curdir, training_dir, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        
    for line in lines:
        source_path = line[0]
        filepath, filename = os.path.split(source_path)
        current_path = os.path.join(os.curdir, training_dir, 'IMG', filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        images.append(np.fliplr(image))
        measurements.append(-measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation

model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))

model.add(Convolution2D(20, 5, 5, border_mode="valid"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Convolution2D(64, 5, 5, border_mode="valid"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, 
    shuffle=True, epochs=5)
model.save('model.h5')

