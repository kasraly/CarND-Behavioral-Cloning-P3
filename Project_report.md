# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./resources/missingLane.jpg "Failed Scene"
[image2]: ./resources/shadow.jpg "Failed Scene"
[image3]: ./resources/figure_1.png "Trainign and Validation Loss"
[image4]: ./resources/steering_filter.png "Filtering Steering Values"
[image5]: ./resources/bridge.jpg "bridge"
[image6]: ./resources/RedStripeCurb.jpg "Curb with red stripes"
[image7]: ./resources/Shadow2.jpg "Tree shadow on the road"
[image8]: ./resources/SideDirt.jpg "Dirt on side of the road"
[image9]: ./resources/center.jpg "Center"
[image10]: ./resources/left.jpg "Left"
[image11]: ./resources/right.jpg "Right"

---
This report is the details about the behavioral cloning project of the Udacity self-driving car program.
I have organized this report based on the items in the ruberic (https://review.udacity.com/#!/rubrics/432/view)

## Required Files

### Submitted files

My project includes the following files: 
* model.py contains the script to create and train the netwrok.
* drive.py, This is mostly the original file provided for the project. I only modified the speed set point to 25, and tuned the gains of the speed PI controller.
* model.h5, a saved copy of trained model that was used for autonomous driving and generating the video clip
* run1.mp4, a recorded video of car driving in autonmous mode using the above model 
* Project_report.md, this report whcih includes details about approach and model and summarizing the results

## Quality of Code

### Code functionality
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track #1 by executing 
```sh
python drive.py model.h5
```

### Code readability

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it includes some comments to explain how the code works.
The code employs a generator to supply the training and validation data and overcome memory limitation issues.

## Model Architecture and Training Strategy

### Model architecture

I utilized the Nvidia model architecture. 5 convolution layers reduce the input size from 3x75x320 to 64x1x32 (model.py lines 113-124), followed by 4 fully connected layers (model.py lines 130-134)
The model includes RELU layers to introduce nonlinearity after each linear layer, expect the final layer which is the output of the model
The input is normalized in the first layer to 0 and 1 (model.py line 109).

### Reducing model overfitting

I randomly divided the images and steering values collected into train and validation with 80% - 20% split. Throughout training the validation loss was monitores and training was halted when validation loss started to increase. 
Also I incorporated two noise layers to reduce the overfitting. An additive noise with zero mean and STD of 0.1 is added to the input layer after normalizing the input (model.py line 110). Additionally a multiplicative noise layer with normal noise is added after last convolution layer (model.py line 128). 

Finally the model was ran through simulator after training as the test environment, and checking the performance for new scenes. 

### Tuning the model parameters

I incorporated adam optimizer and let the algorithm adaptively change the learning rate (model.py line 136).
I utilized 'glorot_normal' for initialization of the weights. 

### Training data

I carefully drove the car in training mode to collect data. I also utilizied left and right images as well as mirrored images to enrich training data. 
For scenes that were less frequent in the track (dirt on one side, red curb markings, shadow on the road, etc.) extra training data was collected in small driving moments. 

Please see section blow on training data for more detail.

## Architecture and Training Documentation

### Solution design

The idea behind behavioral cloning is to predict the steering angle from images by showing the model appropriate steering values for various scenes.
Since the input is an image, convolution layers will extract relevant features from the image, and fully connected layer will translate the features into steering angle. 

First, I started with the LeNet architecture. The only change that I made was to add a layer for preprocessing the image and change the input size from 32x32x3 to 75x320x3 (the cropped image). The preprocessign layer crops the top 65 and  bottom 20 pixels resulting in a 75x320 image. then the input was divided by 255 to normalize the color data to 0 and 1. 
This model was able to achive very low training data loss with few epochs, but validation data loss would start increasing very early in trainign showing signs of overfitting. By terminating the training after evaluation loss was starting to increase, the model was partly fucntional. The car would drive properly in more frequent and less cluttered scence, but would fail in more complex scenes with shadows or lane marks missing on one side. My efforts to add more training data while trying to prevent overfitting proved to be not effective. 
I think the problem with LeNet network is that it is too big while not being deep enoguh. Since our input image is much larger than original LeNet architecture, there are many features after the last convolution layer resulting in farily big network. This would result in a much bigger betwork than what might be required for this task. Also two convolution layers might not be enough to extract the relevant features from the image. 

Next I tried the Nvidia architecture. The Nvidia model consists of 5 convolution layers and 4 fully conenncted layers. I changed the filter size for 4th convolution layer from 3x3 to 4x4. Since the my input image size is different than original Nvidia model, this change would result in a height of 1 at the last convolution layer and keep the model consistent with Nvidia model. After careful training of the model, it would drive the car fairly smooth but still would fail in a few complex scence. see scenes below:

![alt text][image1] ![alt text][image2]

It seemed that the percentage of trainign data for those scences is relatively small compared to total training data. Therefore, the model would not sacrifice these scenes when minimizing the error for the rest of the training data. To overcome this, I identified the scences that were performing poorely and collected training data for those scenes only.

Additionally, to minimize the impact of overfitting, I added two noise layers to the model. I included one additive noise layer at the input layer of the model. and one multiplicative noise layer after last convolution layer and flattening the features. With this addition the overfitting was not as significant as before and training could be performed for more epochs. 

After collecting enough training data of the less frequent scenes, the model was able to succesfully control the car for the track #1 

### Final Model Architecture

The final model architecture is presented in the table below. 
  Layer #  | Operation |Input Size | Output Size
  ------------- | -------------|--|--
  1  | Cropping | 160x320x3| 75x320x3
  2 | normalize to [0, 1] | 75x320x3 | 75x320x3
  3  | Additive normal noise, mean=0, STD= 0.1 | 75x320x3|75x320x3
4  | Conv2D, depth 24, filter: 5x5, strides 2x2, valid padding, ReLU | 75x320x3| 36x158x24
5  | Conv2D, depth 36, filter: 5x5, strides 2x2, valid padding, ReLU | 36x158x24| 16x77x36
6  | Conv2D, depth 48, filter: 5x5, strides 2x2, valid padding, ReLU | 16x77x36| 6x37x48
7  | Conv2D, depth 64, filter: 4x4, strides 1x1, valid padding, ReLU | 6x37x48 | 3x34x64
8  | Conv2D, depth 64, filter: 3x3, strides 1x1, valid padding, ReLU | 3x34x64 | 1x32x64
9  | Flatten and Multiplicaitve Noise, mean = 1, STD ~ 0.5 | 1x32x64 | 2048
10  | Fully connected , ReLU | 2048 | 100
11  | Fully connected , ReLU | 100 | 50
12  | Fully connected , ReLU | 50 | 10
13  | Final Output, Fully connected , no activation | 10 | 1

I employed Adam optimizer for training with Mean Squared Error as the Loss fucntion.
I used 25 Epochs of 100 mini-batch of 64 samples. note that each training epoch only inlcuded a fraction of training data. This was to observe and prevent overfitting. the evaluation was performed on the whole evaluation dataset after each epoch. 

### Training Data Set & Training Process

I started with recording one full lap oftrack #1 in the original direction and another full lap driving track #1 in the reverse direction. I maintained the car in the center of the track and avoided driving too fast. 

I identified the follwoing scenes that are less frequent:
- brige
- curves with red stripes 
- missing side lane marks
- tree shadows 

Then I collected more training data for each scene type to make the ratio of  differnt scenes more equal. Images below show some of the less frequent scenes for whcih more training data was collected. 

![alt text][image5]![alt text][image6]![alt text][image7]![alt text][image8]

I augmented each scene to 6 training sample by incoporating center, left and right images as well as the filpped version of these images. For left and right images, I added a correction of +0.15 and -0.15 to steering, respectively. When flipping the images, I also multiplied the coresponding steering value with -1. Images below, from left to right, show the camera perspective for left, centre and right images

![alt text][image10]![alt text][image9]![alt text][image11]

Since I used the keyboard to control the car during the training, the steering angles were spikes rather than smooth and gradual changes. I filtered the steering values using a moving average filter. I averged the value for 5 frames before and after each images and replaced that with steering value. image below show the raw steering and the filtered one.

![alt text][image4]

After the collection process (considering left, center, righI, and flipped images) I had 16000 data points. I randomly shuffled the data set and put 20% of the data into a validation set. 
During training I employed mini-batches of 64 samples. I trained the model for 25 epochs. The training and validation loss is shown in figure below. Since I used the noise layers, the overfitting was not significant, but the reduction in loss after 25 epochs was minimal.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
