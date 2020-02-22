#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Oct 20 16:01:38 2018

@author: Varun
"""

import numpy as np
import pandas as pd
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import Model
from keras import optimizers
from keras import callbacks
from keras import regularizers
from keras import backend as K
from keras.preprocessing import image
from keras import regularizers


# dimensions of our images.
img_width, img_height = 224, 224
best_weights = "Prediction27.h"
better_weights = "CrazyPrediction35Better.h"
Ultimate_train = 'data/Ultimate_train'
train_data_dir1 = 'data/training_set1'
validation_data_dir1 = 'data/validation_set1'
train_data_dir2 = 'data/training_set2'
validation_data_dir2 = 'data/validation_set2'
train_data_dir3 = 'data/training_set3'
test_data_dir='data/test1/test/'
nb_train_samples = 10820
nb_validation_samples = 4000
epochs = 600
batch_size = 16




vgg16model = applications.VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape = (img_width, img_height, 3))


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.75, zoom_range=0.75, rotation_range=20, horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

Ultimate_traingen1 = train_datagen.flow_from_directory(
    Ultimate_train,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

train_generator1 = train_datagen.flow_from_directory(
    train_data_dir1,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

train_generator2 = train_datagen.flow_from_directory(
    train_data_dir2,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

train_generator3 = train_datagen.flow_from_directory(
    train_data_dir3,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator1 = test_datagen.flow_from_directory(
    validation_data_dir1,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator2 = test_datagen.flow_from_directory(
    validation_data_dir2,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16model.output_shape[1:]))
top_model.add(Dense(256, activation='relu' ))
top_model.add(Dropout(0.85))
top_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs= vgg16model.input, outputs= top_model(vgg16model.output))


print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-13, nesterov = True),
              metrics=['accuracy'])

earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=17, verbose=0, mode='auto', baseline=None)
checkpoint = callbacks.ModelCheckpoint(better_weights, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
model.fit_generator(
    Ultimate_traingen1,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[checkpoint, earlystop],
    validation_data=validation_generator1,
    validation_steps=nb_validation_samples // batch_size)


model.load_weights(better_weights)


predictions = open('data/Crazypredictions35Normal.csv', mode ='w', newline='')
prediction_writer = csv.writer(predictions)

prediction_writer.writerow(['Id','Expected'])

for i in range(7090):
    img = image.load_img(test_data_dir +'test_'+ str(i+1) +'.jpg', target_size = (224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img/255)
    prediction_writer.writerow(['test_' + str(i+1) + '.jpg', result[0][0]])
predictions.close()
    






# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:-5]:
    layer.trainable = False
    
print(model.summary())
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9, decay=1e-13),
              metrics=['accuracy'])


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.75,
    zoom_range=0.75,
    rotation_range=10,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator1 = train_datagen.flow_from_directory(
    train_data_dir1,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

train_generator2 = train_datagen.flow_from_directory(
    train_data_dir2,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

train_generator3 = train_datagen.flow_from_directory(
    train_data_dir3,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir2,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model

epochs = 200

earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=16, verbose=0, mode='auto', baseline=None)
checkpoint = callbacks.ModelCheckpoint(better_weights, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
model.fit_generator(
    Ultimate_traingen1,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[checkpoint, earlystop],
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.load_weights(better_weights)

# predict and save into csv

predictions = open('data/Crazypredictions35FineTuned.csv', mode ='w', newline='')
prediction_writer = csv.writer(predictions)

prediction_writer.writerow(['Id','Expected'])

for i in range(7090):
    img = image.load_img(test_data_dir +'test_'+ str(i+1) +'.jpg', target_size = (224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img/255)
    prediction_writer.writerow(['test_' + str(i+1) + '.jpg', result[0][0]])
predictions.close()
    






