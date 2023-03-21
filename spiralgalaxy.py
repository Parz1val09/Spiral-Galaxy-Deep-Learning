import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras.utils as image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras.optimizers import Adam

import tensorflow as tf

bar = cv2.imread('data/train/bar/3s43jblh4fs51.jpg')
bar = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)
bar = cv2.resize(bar, (150, 150))
bar = bar.astype('float32') / 255.0

nor = cv2.imread('data/train/nor/eso9845d.jpg')
nor = cv2.cvtColor(nor, cv2.COLOR_BGR2RGB)
nor = cv2.resize(nor, (150, 150))
nor = nor.astype('float32') / 255.0

# Define image augmentation generator
image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1/255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

# Define input shape and create model
input_shape = (150,150,3)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
model.summary()

# Train model
batch_size = 16
train_image_gen = image_gen.flow_from_directory('data/train', 
                                                target_size=input_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory('data/test', 
                                                target_size=input_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

epochs = 85;

steps_per_epoch = int(np.ceil(train_image_gen.samples / batch_size))

validation_steps = int(np.ceil(test_image_gen.samples / batch_size))

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
results = model.fit(train_image_gen, epochs=epochs, validation_data=test_image_gen,steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks = [tensorboard_callback])

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# Convert the array to a pandas dataframe and transpose it
df1 = pd.DataFrame({'data': results.history['acc']})
df2 = pd.DataFrame({'data': results.history['loss']})
df3 = pd.DataFrame({'data': results.history['val_acc']})
df4 = pd.DataFrame({'data': results.history['val_loss']})

# Write the dataframe to a CSV file
df1.to_excel('acc.xlsx', index=False, header=False)
df2.to_excel('loss.xlsx', index=False, header=False)
df3.to_excel('valacc.xlsx', index=False, header=False)
df4.to_excel('valloss.xlsx', index=False, header=False)

from keras.models import load_model
import os
model.save(os.path.join('models','bar_nor_100epochs.h5'))
new_model = load_model(os.path.join('models', 'bar_nor_100epochs.h5'))

folder_path = 'data/test/bar/'

# Get a list of all the file names in the folder
file_names = os.listdir(folder_path)

# Initialize an empty list to store the image arrays
images = []

# Loop over each file name in the list
for file_name in file_names:
    # Construct the full path to the file
    file_path = os.path.join(folder_path, file_name)
    
    # Load the image and resize it to (150, 150)
    img = image.load_img(file_path, target_size=(150, 150))
    
    # Convert the image to a numpy array and normalize it
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Append the image array to the list
    images.append(img_array)
    
# Concatenate the list of image arrays into a single numpy array
images = np.concatenate(images, axis=0)

# Predict the output for the array of images using your model
yhatbar = model.predict(images)

print(np.round(yhatbar))

folder_path = 'data/test/nor/'

# Get a list of all the file names in the folder
file_names = os.listdir(folder_path)

# Initialize an empty list to store the image arrays
images = []

# Loop over each file name in the list
for file_name in file_names:
    # Construct the full path to the file
    file_path = os.path.join(folder_path, file_name)
    
    # Load the image and resize it to (150, 150)
    img = image.load_img(file_path, target_size=(150, 150))
    
    # Convert the image to a numpy array and normalize it
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Append the image array to the list
    images.append(img_array)
    
# Concatenate the list of image arrays into a single numpy array
images = np.concatenate(images, axis=0)

# Predict the output for the array of images using your model
yhatnor = model.predict(images)

print(np.round(yhatnor))
    
print(train_image_gen.class_indices)


