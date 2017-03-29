from keras.models import Sequential, Model
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
#import sklearn
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import os


### Load Data and Images
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
# Using Multipule Cameras
with open(csv_file, 'r') as f:
  reader = csv.reader(f)
      for row in reader:
          steering_center = float(row[3])

          # create adjusted steering measurements for the side camera images
          correction = 0.2 #TODO: tune
          steering_left = steering_center + correction
          steering_right = steering_center - correction

          # read in images from center, left and right cameras
          directory = ".data/"
          img_center = process_image(np.asarray(Image.open(path + row[0])) 
          img_left = process_image(np.asarray(Image.open(path + row[1])))
          img_right = process_image(np.asarray(Image.open(path + row[2])))

          # add images and angles to data set
          car_images.extend(img_center, img_left, img_right)
          steering_angles.extend(steering_center, steering_left, steering_right)
  
# Data Augmentation
image_flipped = np.fliplr(image)
measurement_flipped = -measurement


### Generator !? 
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


### Training: NVIDIA Architecture
# ref: images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf                               

# Trimmed image format
ch, row, col = 3, 80, 320  

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# set up lambda layer
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(ch, row, col), output_shape=(ch, row, col)))
# set up cropping2D layer
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# ...


### Test and Evaluate !?
model.compile(loss='mse', optimizer='adam')
#model.fit_generator(train_generator, samples_per_epoch= /
#    len(train_samples), validation_data=validation_generator, /
#    nb_val_samples=len(validation_samples), nb_epoch=3)

history_object = model.fit_generator(
    train_generator, 
    samples_per_epoch=len(train_samples), 
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples), 
    nb_epoch=5, 
    verbose=1
)

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
