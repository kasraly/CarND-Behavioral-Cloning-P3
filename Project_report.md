# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./resources/missingLane.jpg "Failed Scene"
[image2]: ./resources/shadow.jpg "Failed Scene"
[image3]: ./resources/figure_1.png "Trainign and Validation Loss"
[image4]: ./resources/steering_filter.png "Filtering Steering Values"

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

### Dolution design

The idea behind behavioral cloning is to predict the steering angle from images by showing the model appropriate steering values for various scenes.
Since the input is an image, convolution layers will extract relevant features from the image, and fully connected layer will translate the features into steering angle. 

First, I started with the LeNet architecture. The only change that I made was to add a layer for preprocessing the image and change the input size from 32x32x3 to 75x320x3 (the cropped image). The preprocessign layer crops the top 65 and  bottom 20 pixels resulting in a 75x320 image. then the input was divided by 255 to normalize the color data to 0 and 1. 
This model was able to achive very low training data loss with few epochs, but validation data loss would start increasing very early in trainign showing signs of overfitting. By terminating the training after evaluation loss was starting to increase, the model was partly fucntional. The car would drive properly in more frequent and less cluttered scence, but would fail in more complex scenes with shadows or lane marks missing on one side. My efforts to add more training data while trying to prevent overfitting proved to be not effective. 
I think the problem with LeNet network is that it is too big while not being deep enoguh. Since our input image is much larger than original LeNet architecture, there are many features after the last convolution layer resulting in farily big network. This would result in a much bigger betwork than what might be required for this task. Also two convolution layers might not be enough to extract the relevant features from the image. 

Next I tried the Nvidia architecture. The Nvidia model consists of 5 convolution layers and 4 fully conenncted layers. I changed the filter size for 4th convolution layer from 3x3 to 4x4. Since the my input image size is different than original Nvidia model, this change would result in a height of 1 at the last convolution layer and keep the model consistent with Nvidia model. After careful training of the model, it would drive the car fairly smooth but still would fail in a few complex scence. see scenes below:
![alt text][image1] ![alt text][image2]
It seemed that the percentage of trainign data for those scences is relatively small


Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

### 3. Creation of the Training Set & Training Process

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
