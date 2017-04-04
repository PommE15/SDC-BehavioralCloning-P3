# Writeup report of Behavioral Cloning
This writeup is based on the [rubric points](https://review.udacity.com/#!/rubrics/432/view) and 
this [template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md).

[//]: # (Image References)
[figure1]: ./assets/cropped2d.jpg ""
[figure2]: ./assets/results.png ""

 
## Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) - containing the script to create and train the model
* [drive.py](./drive.py) - for driving the car in autonomous mode
* [model.h5](./model.h5) - containing a trained convolution neural network 
* [writeup_report.md](./writeup_report.md) - summarizing the results
* [video.mp4](./video.mp4) - recording one lap of my vehicle driving autonomously on the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and the `drive.py` file, the car can be driven autonomously at speed of 16mph around the track one by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code of my experiments, training, and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


## Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy was to keep the model simple, make sure it's working, and then improve it.

My first step was to: 
* use only center images with steering data, 
* apply normailzation and cropping on those images,
* trian a simple one layer convolution neural network (cnn) for an epoche, and 
* check if the model can predict the steering directions. 

I assumed a one layer cnn model might work because the way how I cropped the images (see point 3 below) leaves them with simple geometry patterns for training. To see if this model is effective, I picked three images with straight, left, right steering each for a quick testing <sup>[2.11]</sup>.

![alt text][figure2]

The last three lines in the figure above output the predict results (first column) and the steering angles (second column) of the three images I picked. After training a model and testing it on the images, I checked if the numbers were properly prediected. For example, the second line shows the left steering image and it's expected to have a negative number. Compare with the other two images, the number is also expected to be larger than the other two. The straight steering image, on the other hand, is expected to be a number even closer to zero. If these basic checkings were passed, I ran the model in the simulator and see if the vehicle was able to drive autonomously around the track without leaving the road for at least a lap. 

Once I have a working model, I experiemented with other additional steps:
* add more data process we learned in the class, 
* play with different cnn model (including NVIDIA Architecture <sup>[1]</sup>), and 
* tune parameters (see point 5 below).

In all steps, I kept an eye on the loss (mean squared error) on both training and validation set to monitor the overfitting. Among all the model modification and tuning, many works but a simple model stands out. 

#### 2. Model Architecture

My final model consists of:

| `@model.py` | layers     | detail |
|-------------|------------|--------|
| `line 77`   | Cropping2D | 66px from top and 30px from bottom |
| `line 80`   | Lambda     | zero-centered normailzation |
| `line 83`   | Conv2D     | depth 9, filter sizes 5x5, stride of 2, and ReLU |

The model includes a ReLU activation function in the Convolutional layer to introduce nonlinearity, and the data is also cropped and normalized in the model using Keras layers.

#### 3. Everything about data in this project

#### Training set:
I used center images that provids in the class and flipped them (and their angles) in order to double the amount and create more samples for right turns. Eventually, I didn't choose to use images from left and right cameras as my experiments didn't show improvement by using them. 

* source: center images
* data augmentation: double samples with flipped center images
* multiple cameras: add left and right (discarded)

After the collection process, I then preprocessed this data by converting it's color space.

#### Data preprocess:
* convert from BGR -> RGB <sup>[3.5]</sup>
* corp (in Keras)
* normalize (in Keras)

![alt text][figure1]

The image was cropped 66px from the top and 30px from the bottom to keep just the area of the road.

#### Training and validation sets
* shuffle
* 80/20 data split for training and validation sets `@model.py line 24`

Finally, I randomly shuffled the data set and put 20% of the data into a validation set. 

#### 4. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The ideal number of epochs was between 3 to 7 based on the results of my experiments. The validation set is then used to help determine if the model was over or under fitting. For example, the dropout layer was discared as the validation error indicated under fitting. 

#### 5. Model parameter tuning

* filter depths: 9, 12, 16, 24, 36 in the convolutional layer, 9 performs the best
* epochs: between 3 to 7, 5 is better in avarage
* optimizer: adam, learning rate was not tuned manually `@model.py line 108` 


## Simulation Result

In the simulation, the car is always driving autonomously on the road around track one up to 16mph. `video.mp4`


## Reference
1. [End to end learning for self-driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars), NVIDIA
2. [Behavioral cloning cheatsheet](https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet)
3. [Behavioral cloning non-spoiler hints](https://discussions.udacity.com/t/behavioral-cloning-non-spoiler-hints/233194)
