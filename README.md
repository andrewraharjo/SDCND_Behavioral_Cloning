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


