#**Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
#Files Submitted & Code Quality

##1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files: 
* model.py contains the script to create and train the netwrok.
* drive.py, I did not modify this file except changing the speed set point to 25, and tuning the gains of the speed PI controller.
* model.h5, a saved copy of trained model used autonmous driving and generating the mvideo clip
* run1.mp4, a recorded video of car driving in autonmous mode using the above model 
* writeup_report.md, the projet report, providign details into approach and model and summarizing the results

##2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

##3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it includes some comments to explain how the code works.

#Model Architecture and Training Strategy

##1. An appropriate model architecture has been employed

I utilized the Nvidia model architecture. 5 convolution layers reduce the input size from 3x75x320 to 64x1x32 (model.py lines 113-124). followed by 4 fully connected layer (model.py lines 130-134)
The model includes RELU layers to introduce nonlinearity after each linear layer expect the final output 
I normalized the input data to 0 and 1 (model.py line 109).

##2. Attempts to reduce overfitting in the model

I randomly divided the images and steering values collected into train and validation with 80% - 20% split. Throughout training the validation loss was monitores and training was halted when validation loss started to increase. 
Also I incorporated two noise layers to counteract with overfitting. An additive noise with zero mean and STD of 0.1 is added to the input layer (model.py line 110). additionally a multiplicative noise layer of normal noise with mean 1 and STD of around 0.5 as added after last convolution layer (model.py line 128). 

Finally the model was ran through simulator to as the test environment. 

##3. Model parameter tuning

I incorporated adam optimizer and let the algorithm adaptively change the learning rate (model.py line 136).

##4. Appropriate training data

I carefully drove the car in training mode in the center lane to collect data. I also utilizied left and right images as well as mirrored images to enrich training data. 
For scenes that were less frequent in the track (dirt on one side, red curb markings, shadow on the road, etc.) extra training was collected in small driving moments. 

Please see section blow on training data for more detail.

#Model Architecture and Training Strategy

##1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
