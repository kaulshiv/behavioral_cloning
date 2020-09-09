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
[image3]: ./writeup_images/close_to_left.jpg "left"
[image4]: ./writeup_images/close_to_right.jpg "right"
[image5]: ./writeup_images/center.jpg "center"



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

My model is more or less identical to the model used by Nvidia to in "End to End Learning for Self-Driving Cars." The network diagram can be seen below. Each of the convolutional layers have a ReLu nonlinearity between them. My network differs in that it replaces the last dense layer with a layer of 1 neuron. This is necessary since the network should provide a single value corresponding to the steering angle. 

![nvidia network][image2]

My network also adds a cropping layer after the normalization phase, where it takes out the top 70 rows of pixels and the bottom 25 rows. This gives an input shape of `(65,320,3)` to the convolutional part of the network. I found cropping to be helpful since it removes the sky and hood of the car from the training images. Those are features that aren't intuitively useful for learning the correct steering angle. I normalized each image by scaling each pixel between -0.5 and 0.5.


#### 2. Generalization

Although the model overfits slightly on the training set as shown in the graph below, I did not find it necessary to add dropout layers. The data augmentation I performed was sufficient regularization as it allowed me to drive around the track safely without any problems (see [run2.mp4](run2.mp4)). My augmentation strategy is discussed in detail below.
![loss][image1]


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Creation of the Training Set & Training Process

Most of the data provided by udacity was center of the lane driving, displayed below. This is beneficial in the "normal" mode of driving, but the model never learns how to react when it comes too close to an edge if it is only exposed to this type of data.

![center][image5]


To remedy this, most of the data I recorded involved correcting for when the vehicle was too close to one side of the lane like in the two images below.
![left][image3]
![right][image4]


I also collected data driving both clockwise and counter-clockwise around the track. After accumulating all of my own images with Udacity's images, I performed a final augmentation steps programmatically (see lines 95-109 in `model.py`). These involved flipping each image over the vertical axis, then including images from the right and left cameras (and flipping those images as well). All together, this produced a 6x increase in the data.

I used a training-validation split of `80`-`20`. The final size of the training dataset after augmentation was `74304` images, while the validation set was `18582` images. Each image had dimensions `(160,320,3)`.


#### 5. Solution Design Approach

The overall strategy for deriving a model architecture was to start with NVIDIA's baseline model for behavioral cloning and iteratively test to see how to get a model that would perform well on the simulated track.

I first tested the model with only Udacity's provided data. The model's loss during training increased for both the validation and training sets. It also drove straight into the first curve onto the simulator. My conclusion was the model simply didn't have enough data to learn the task properly.

As a result, I tried my first programmatic data augmentation techniques discussed above. The advantage of this was I didn't have to spend time collecting my own data. The model's MSE loss improved on both the training and validation set. It also performed decently well on the track, but still had a lingering issue with tight curves.

In order to fix this, I had to finally bite the bullet and collect my own data using the simulator. I found it incredibly helpful to download the simulator onto my own machine instead of using the provided environment, so I didn't have to needlessly waste GPU time. I cloned [this](https://github.com/endymioncheung/CarND-MacCatalinaSimulator) repo to get the simulator.

The data I collected was primarily "recovery" data, which would teach the network when it was unsafely close to an edge and how to properly change the steering angle to compensate. The final results were excellent and can be viewed in `run2.mp4` included in this repo.