from keras.models import Sequential, Model
from keras.layers.core import Lambda, Dropout, Flatten, Dense
from keras.layers.convolutional import Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#import sklearn
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import os


### Load Data and Images
samples = []
csv_file = './data/driving_log.csv'
with open(csv_file) as f:
    reader = csv.reader(f)
    # skip the headers
    next(reader, None)  
    for line in reader:
        samples.append(line)


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
                # Using multipule cameras
                
                # TODO: tune
                correction = 0.1
                
                path = './data/'
                image_center = cv2.imread(path + batch_sample[0])
                image_left   = cv2.imread(path + batch_sample[1].strip())
                image_right  = cv2.imread(path + batch_sample[2].strip())
                angle_center = float(batch_sample[3])
                angle_left   = angle_center + correction
                angle_right  = angle_center - correction
                
                # Data Augmentation
                '''
                #image_flipped = np.fliplr(image)
                #measurement_flipped = -measurement
                '''

                images.extend([image_center, image_left, image_right])
                angles.extend([angle_center, angle_left, angle_right])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


### Training: NVIDIA Architecture
# ref: images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf                               

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# set up cropping2D layer
model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(160, 320, 3)))

# RGB -> YUV
# cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# ...

# set up lambda layer
model.add(Lambda(lambda x: x/255 - 0.5 ))

# conv
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, 5, 5, subsample=(0, 0), activation='relu'))
model.add(Conv2D(64, 5, 5, subsample=(0, 0), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# fc
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))

# output
model.add(Dense(1, activation='sigmoid'))


### Test and Evaluate !?
model.compile(loss='mse', optimizer='adam')
model.fit_generator(
    train_generator, 
    samples_per_epoch=len(train_samples), 
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples), 
    nb_epoch=1
)

# save model
model.save('model.h5')

'''
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
'''
