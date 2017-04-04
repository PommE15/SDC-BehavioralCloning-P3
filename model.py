from keras.models import Sequential, Model
from keras.layers.core import Lambda, Dropout, Flatten, Dense
from keras.layers.convolutional import Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2

# Parameter Tuning
TEST_RATE   = 0.2
BATCH_SIZE  = 32
CORRECTION  = 0.2
CROP_TOP    = 66
CROP_BOTTOM = 30 
CONV_FDEPTH = 9
CONV_FSIZE  = 5
CONV_STRIDE = 2
KEEP_RATE   = 0.9 
EPOCH       = 5


### Load Data and Image
samples = []
csv_file = './data/driving_log.csv'
with open(csv_file) as f:
    reader = csv.reader(f)
    next(reader, None) # skip the headers
    for line in reader:
        samples.append(line)

# split into training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=TEST_RATE)


### Generator 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                path = './data/'
                
                ### BGR -> RGB / YUV
                image_origin = cv2.imread(path + batch_sample[0])
                image_center = cv2.cvtColor(np.copy(image_origin), cv2.COLOR_BGR2RGB) # / cv2.COLOR_BGR2YUV
                angle_center = float(batch_sample[3])
                
                #### Data Augmentation: flip
                image_flipped_center = np.fliplr(np.copy(image_center))
                
                ### Using multipule cameras
                '''
                correction = CORRECTION
                image_left  = cv2.imread(path + batch_sample[1].strip())
                image_right = cv2.imread(path + batch_sample[2].strip())
                angle_left  = angle_center + correction
                angle_right = angle_center - correction
                #image_flipped_left  = np.fliplr(image_left)  
                #image_flipped_right = np.fliplr(image_right) 
                '''
 
                images.extend([image_center, image_flipped_center])#, image_left, image_right, image_flipped_left, image_flipped_right])
                angles.extend([angle_center, -angle_center])#, angle_left, angle_right, -angle_left, -angle_right])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


### Training
model = Sequential()

# cropping
model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0, 0)), input_shape=(160, 320, 3)))

# lambda: zero-centered with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5))

# conv with relu
model.add(Conv2D(CONV_FDEPTH, CONV_FSIZE, CONV_FSIZE, subsample=(CONV_STRIDE, CONV_STRIDE), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(KEEP_RATE))

# fc / output
model.add(Flatten())
model.add(Dense(1))

# NVIDIA Architecture
# ref: images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf                               
'''
# conv
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(32, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
# fc
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(3, activation="relu"))
'''


### Validate, Test, and Save Model
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(
    train_generator, 
    samples_per_epoch=len(train_samples), 
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples), 
    nb_epoch=EPOCH,
    #verbose= 1
)
'''
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

model.save('model_candidate.h5') # model.h5
print("model is saved")


### Quick Evaluation
image1 = cv2.cvtColor(cv2.imread('./data/IMG/center_2016_12_01_13_30_48_287.jpg'), cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(cv2.imread('./data/IMG/center_2016_12_01_13_46_29_398.jpg'), cv2.COLOR_BGR2RGB)
image3 = cv2.cvtColor(cv2.imread('./data/IMG/center_2016_12_01_13_33_08_548.jpg'), cv2.COLOR_BGR2RGB)
angle1 = float(model.predict(image1[None, :, :, :], batch_size=1))
angle2 = float(model.predict(image2[None, :, :, :], batch_size=1))
angle3 = float(model.predict(image3[None, :, :, :], batch_size=1))
print("{:.6f} {:.6f}".format(angle1, 0))
print("{:.6f} {:.6f}".format(angle2, -0.2971161))
print("{:.6f} {:.6f}".format(angle3, 0.2148564))
