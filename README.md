# **Behavioral Cloning** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/loss.png "loss"
[image2]: ./writeup_images/nvidia_network.png "network"
[image3]: ./writeup_images/close_to_left.png "left"
[image4]: ./writeup_images/close_to_right.png "right"



---
## **Files Submitted & Code Quality**

#### 1. Submission files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode. I changed the provided script slightly to match pass an BGR rather than a RGB formatted image into the model for inference.
* [model.h5](model.h5) model trained using a combination of the data provided by Udacity and data I generated 
* [writeup_report.md](writeup_report.md) (this file) summary of results
* [run2.mp4](run2.mp4) video of the car running in autonomous mode on Track 1 in the Udacity simulator using my model.

#### 2. Command to train model and run model

To generate a new saved model, I simply ran.
```sh
python model.py
```

In order to view how the model performs, I ran this command while the simulator was running
```sh
python drive.py model.h5
```

#### 3.Code documentation

View the docstrings in model.py to learn how I trained an saved the models.

---

## **Model Architecture and Training Strategy**

#### 1. Architechture

My model is more or less identical to the model used by Nvidia to in "End to End Learning for Self-Driving Cars." The network diagram can be seen below. Each of the convolutional layers have a ReLu nonlinearity between them. My network differs in that it replaces the last dense layer with a layer of 1 neuron. This is necessary since the network should provide a single value corresponding to the steering angle. My network also adds a cropping layer after the normalization phase, where it takes out the top 70 rows of pixels and the bottom 25 rows. This gives an input shape of `(65,320,3)` to the convolutional part of the network. I found cropping to be helpful since it removes the sky and hood of the car from the training images. Those are features that aren't intuitively useful for learning the correct steering angle.

![nvidia network][image2]



#### 2. Generalization

Although the model overfits slightly on the training set as shown in the graph below, I did not find it necessary to add dropout layers. The data augmentation I performed was sufficient regularization as it allowed me to drive around the track without any problems (see [run2.mp4](run2.mp4)). My augmentation strategy is discussed in detail below.
![loss][image1]


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 5. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


 My augmentation steps included collecting data driving both clockwise and counter-clockwise around the track. I also was sure to include some drastic "corrections" where I would drive the car back to the center of the lane after it had gotten too close to one side of the track.
