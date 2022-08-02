# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:40:35 2022

@author: prans
"""
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPool2D())

# Adding a second Convolution layer
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(MaxPool2D())

# Step 3 - Flatten
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'C:\Users\prans\Downloads\P16-Convolutional-Neural-Networks\Part 2 - Convolutional Neural Networks\dataset\training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'C:\Users\prans\Downloads\P16-Convolutional-Neural-Networks\Part 2 - Convolutional Neural Networks\dataset\test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit(training_set,
               epochs = 10,
               validation_data = test_set)

# Making new prediction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img(r'C:\Users\prans\Downloads\P16-Convolutional-Neural-Networks\Part 2 - Convolutional Neural Networks\dataset\single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print('Prediction = ', prediction)
