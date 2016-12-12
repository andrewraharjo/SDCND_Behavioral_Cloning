# Self-Driving Car Engineer Nanodegree
# Deep Learning - Transfer Learning
## Project: Deep Learning with Keras - Run the simulator to drive the car itself

### Overview
We’ve created a simulator for you based on the Unity engine that uses real game physics to create a close approximation to real driving.

### Running the Simulator

Once you’ve downloaded it, extract it and run it.

When you first run the simulator, you’ll see a configuration screen asking what size and graphical quality you would like. We suggest running at the smallest size and the fastest graphical quality. We also suggest closing most other applications (especially graphically intensive applications) on your computer, so that your machine can devote its resource to running the simulator.

### Collecting Training Data

In order to start collecting training data, you'll need to do the following:

    Enter Training Mode in the simulator.
    Start driving the car to get a feel for the controls.
    When you are ready, hit the record button in the top right to start recording.
    Continue driving for a few laps or till you feel like you have enough data.
    Hit the record button in the top right again to stop recording.

If everything wen't correctly, you should see the following in the directory you selected:

    IMG folder - this folder contains all the frames of your driving.
    driving_log.csv - each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of your car. You'll mainly be using the steering angle.

Now that you have training data, it’s time to build and train your network!

Use Keras to train a network to do the following:

    Take in an image from the center camera of the car. This is the input to your neural network.
    Output a new steering angle for the car.

You don’t have to worry about the throttle for this project, that will be set for you.

Save your model architecture as model.json, and the weights as model.h5.

### Validating Your Network

You can validate your model by launching the simulator and entering autonomous mode.

The car will just sit there until your Python server connects to it and provides it steering angles. Here’s how you start your Python server:

    Install Python Dependencies (conda install …)
        - numpy
        - flask-socketio
        - eventlet
        - pillow
        - keras
        - h5py
    Download drive.py.
    Run Server
        python drive.py model.json
        
Once the model is up and running in drive.py, you should see the car move around (and hopefully not off) the track!

### Recovery
If you drive and record normal laps around the track, even if you record a lot of them, it might not be enough to train your model to drive properly.

Here’s the problem: if your training data is all focused on driving down the middle of the road, your model won’t ever learn what to do if it gets off to the side of the road. And probably when you run your model to predict steering angles, things won’t go perfectly and the car will wander off to the side of the road.

So we need to teach the car what to do when it’s off on the side of the road.

One approach might be to constantly wander off to the side of the road and then wander back to the middle.

That’s not great, though, because now your model is just learning to wander all over the road.

A better approach is to only record data when the car is driving from the side of the road back toward the center line.

So as the human driver, you’re still weaving back and forth between the middle of the road and the shoulder, but you need to turn off data recording when you weave out to the side, and turn it back on when you weave back to the middle.

It’s probably enough to drive a couple of laps of nice, centerline driving, and then add one lap weaving out to the right and recovering, and another lap weaving out to the left and recovering.

### Problems

A common problem in Deep Learning is overfitting on a given training set. In this project, you might overfit by simply memorizing the images and their corresponding steering angles. In order to help you combat this, we've provided you two track to train and test with. An excellent solution would be one that is able to work on both tracks.

There are some common techniques you should consider using to prevent overfitting. These include:

    Adding dropout layers to your network.
    Splitting your dataset into a training set and a validation set.

## Architecture and Training Docs

### Model Design
The way I approached this by resizing the image to the specific needs. I tried to eliminate the unnecessary data and focus on the road. 

I reduced the size of the image by **25%** and used the red channel of YUV Image. It is more efficient in training terms of time and space. The resized image is saved as **features** and synced the data of steering angels as **labels**. I splitted the data into **train** and **validation**, and saved them as **camera.pickle** file

Preprocessing:
- 1.  Resize the image from 320x160 to 80x18.
- 2.  Convert the 80x18 image from RGB to YUV.
- 3.  Select the Red channel only from YUV for preprocessing.
- 4.  Flatten the last 18 Y-axis from the bottom so the new resized image is now 80x18 YUV Red Ch.

I have total 19221 items each contained three images from different angles: center, left, and right. So, there are total 19221 x 3 = 57663 images I reshaped and used for training.

## Training

| Layer (type) | Output Shape | Param # | Connected to |
| :--- | :--- | ---: | :--- |
| convolution2d_1 (Convolution2D) | (None, 16, 78, 16) | 160 | convolution2d_input_1[0][0]  |
| activation_1 (Activation) | (None, 16, 78, 16)| 0 | convolution2d_1[0][0] |
| convolution2d_2 (Convolution2D) | (None, 14, 76, 8)| 1160 | activation_1[0][0]  |
| activation_2 (Activation) | (None, 14, 76, 8) | 0 | convolution2d_2[0][0] |
| activation_3 (Activation)| (None, 12, 74, 4)| 0 | convolution2d_3[0][0] |
| convolution2d_4 (Convolution2D)|(None, 10, 72, 2)|74| activation_3[0][0] |
| activation_4 (Activation)| (None, 10, 72, 2)| 0| convolution2d_4[0][0] |
| maxpooling2d_1 (MaxPooling2D)|(None, 5, 36, 2)|0|activation_4[0][0]               
| dropout_1 (Dropout)|(None, 5, 36, 2)|0|maxpooling2d_1[0][0]             
| flatten_1 (Flatten)|(None, 360)|0|dropout_1[0][0]                  
| dense_1 (Dense)|(None, 16)|5776|flatten_1[0][0]                  
| activation_5 (Activation)|(None, 16)|0|dense_1[0][0]                    
| dense_2 (Dense)|(None, 16)| 272|activation_5[0][0]               
| activation_6 (Activation)|(None, 16)|0|dense_2[0][0]                    
| dense_3 (Dense)|(None, 16)| 272|activation_6[0][0]               
| activation_7 (Activation)|(None, 16)|0|dense_3[0][0]                    
| dropout_2 (Dropout)|(None, 16)|0|activation_7[0][0]               
| dense_4 (Dense)|(None, 1)|17|dropout_2[0][0] 

Total params: 8023

In the CNN architecture, I use Keras builtin support for the Adam optimizer similar on the lecture for Traffic Sign Classifier. The Adam optimizer, as explained in project 2, is Kingma and Ba's modified version of the Stochastic Gradient Descent that allows the use of larger step sizes without fine tuning.  In general, the Adam optimizer uses cross entropy calculations to minimize loss (average distance to the target label in the solution space) and use gradient descent, an iterative optimization technique and algorithm to achieve this goal. Even though using the Adam optimizer should allow us to use larger step sizes (learning rates)/ This made it so that the model never seem to converge, so it never over-fit; however, in subsequent tests, the model performed exceptionally well in steering the car in the simulator and making sure that the car remaining in the center of the lane. 

### Summary
The red channel of the image contains the better information for identifying the road and lanes than green and blue channels. As a result, the size of the image was 18 x 80 x 1.

In my model, I used 4 ConvNet with 1 max pooling layer, and 3 more dense layers after flatten the matrix. For each convolutional layer, I decreased the channel size by half. When the size of the channel became 2 in the fourth convolutional layer, I applied max pooling with dropout with 25%. After flatten the matrix, the size of features became 360. I used dense layers with 16 features 4 times. Each epoch took about 100 seconds and I used 10 epoches to train the data. As a result, the car drove by itself without popping onto the edges or out of the edges.

The interesting thing I noticed was even though the model allowed the car to drive itself, the accuracy was only about 58%. So the accuracy did not have to be high for car to drive autonomously. I believe that to increase the accuracy, I would need more data set and more epoches.
