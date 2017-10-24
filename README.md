**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run2.mp4 for a first person video of the car driving autonomously
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128.

The model includes RELU layers to the convolution layes and ELU to the dense layers to introduce nonlinearity , and the data is normalized in the model using 2 Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, recovering from the left and right sides of the road and driving in both directions of the track.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement an already working architecture.
I even considered using transfer learning, but I decided against it, as I wanted to see how far I can go with my own training.

My first step was to use a convolution neural network model similar to the one implemented by Nvidia.
I thought this model might be appropriate because it had a proven track record and if there were any problems,
I knew they would be comming from the data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set but an even higher mean squared error on the validation set. This implied that the model was overfitting and had a bias, specifically a bias for going straight. 

To combat the bias, I used all the images, not only the ones from the center and I played around with data augmentation.
To combat the overfitting, I modified the model by adding some regularization.
I did this by adding Dropout layers after every fully connected(dense) layer, each with keep probability of 50%.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes :

lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
lambda_2 (Lambda)                (None, 64, 64, 3)     0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 30, 30, 24)    1824        lambda_2[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 13, 13, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 5, 48)      43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 3, 64)      27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 1, 64)      36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 64)            0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 16896)         1098240     flatten_1[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 16896)         0           dense_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 16896)         0           elu_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           1689700     dropout_1[0][0]
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 100)           0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        elu_2[0][0]
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 50)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             51          elu_3[0][0]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovering and to reduce its bias towards driving straight.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would double my training data, I also used the Udacity data.

I then preprocessed this data by by adding random brightness and some warping.

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by the fact that the validation loss kept lowering in each epoch.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
