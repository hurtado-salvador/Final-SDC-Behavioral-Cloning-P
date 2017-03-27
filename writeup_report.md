#Behavioral Cloning Project
###The goals / steps of this project are the following:

Use the simulator to collect data of good driving behavior
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road
Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/3cams_1.jpg "Cameras"
[image2]: ./images/angulos_1.png "Angles"
[image3]: ./images/cams.jpg "Cameras view"
[image4]: ./images/hist_orig.jpg "Histogram Xbox controller"
[image5]: ./images/histograma_2.jpg "Histograma training data"
[image6]: ./images/nvidia1.png "Nvidia Architecture"

#RUBRIC POINTS
###Required Files:
####1.- My project includes the following files:
* model.py containing the script to create and train the model (64x64 px images)
* drive.py for driving the car in autonomous mode (64x64 px images)
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

###Quality of Code:
####2.- Submission includes functional code Using the Udacity provided simulator and my drive.py file, 
the car can be driven autonomously around the track by executing:
python drive.py model.h5

####3.- Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments 
to explain how the code works.

###Model architecture and training strategy:
####4.- Has an appropriate model architecture been employed for the task?
Base on the information of the NVIDIA End To End Deep Learning for Self-Driving Cars.
  https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
The model is as follows.
Layer Type	      Keras Layer
Normalization	    model.add(Lambda(lambda x: x/255 -0.5, input_shape=(64,64,3)))
Cropping	        model.add(Cropping2D(cropping=((10,0),(0,0))))
Convolution	      model.add(Convolution2D(24,5,5, activation='relu', subsample=(2,2), border_mode='same'))
Convolution	      model.add(Convolution2D(36,5,5,activation='relu', subsample=(2,2), border_mode='same'))
Convolution	      model.add(Convolution2D(48,5,5,activation='relu',subsample=(2,2), border_mode='same'))
Convolution	      model.add(Convolution2D(64,3,3,activation='relu'))
Dropout	          model.add(Dropout(0.5))
Convolution	      model.add(Convolution2D(64,3,3,activation='relu'))
Flatten	          model.add(Flatten())
Fully connected	  model.add(Dense(1164))
	                model.add(ELU())
Fully connected	  model.add(Dense(100))
	                model.add(ELU())
Fully connected	  model.add(Dense(50))
	                model.add(ELU())
Fully connected	  model.add(Dense(10))
	                model.add(ELU())
Fully connected	  model.add(Dense(1))

####5.- Has an attempt been made to reduce overfitting of the model?
Dropout layer was added after the 4th Convolution Layer with a drop rate of 50%

####6.- Have the model parameters been tuned appropriately?
Learning rate was adjusted, using 'adam' learning rate of 0.0001, and during testing trying different values from 0.01, 0.001 to 0.0001 
Other parameters tuned include the batch size for the generator function, the samples per epochs and the number of epochs. 

####7.- Is the training data chosen appropriately?
In this matters, originally the training data was generated using a Xbox controller, the problemas observed is that most of the time the values were 0, in order to get more useful continuos values and after testing with the mouse, the training data generated using the mouse was choosed.

![image4]![image5]


The values of the angles were plotted on histogram for comparation, based on that 2 strategies were implemented.
a) For the training 3 times the samples with steeering angle above +/-0.20 (+/- 5°) was added to the training set, and 21 times the samples with steering angle above +/-0.35 (+/-8.75°).
function name "def augment_images(image_name, angles_val):"


b) The use of the left and right images, with a variable angle correction strategy.
In the Step by Step solution video by David Silver https://www.youtube.com/watch?v=rpxZ87YFg0M&index=3&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P&t=6s.
The correction angle for left and right images is defined as a fixed number that can be tuned.

###My implementarion consider the following.
####1.- The angle is the most important value, due the model take this for the optimization.
####2.- The angle for the left and right camera is not fixed it depend on the position of the camera with respect to the center camera.
####3.- The speed of the vehicle also has impact on the steering angle.
Assumptions:
*Left camera is at 0.805mts from the center in the same plane.
*Right camera is at 0.826mts from the center in the same plane.
This information was shared by @anguyen in slack channel.

-Function "def correction_factor(x, sp):"  using as paramters the angle of the center camera and the speed calculates and retuns  the correction factor for left image and right image.
-The computation consider a triangle formed by the forward vector calculated as the distance traveled in 1 second of the giving speed, this is consider as the adjacent side of the triangle.
-The distance from the center of the camera to the left and right camera are considered as the oposite side of the traingle. and variate considering the center vector angle.
#Then the formula to calculate the angle for correction left and righ side is as follows.
Left andgle in degrees is the inverse tangent of divide the oposite side to the forward distance.
"left_angle_grad = degrees(atan(oposite_left / forward_distance))"

![alt text][image3]
![alt text][image2]


###Architecture and Training Documentation:
####8.- Is the solution design documented?
There are 2 mayor challenges in order to train a model from images.
a) Computation needs for the model. Using a Core I3 processor with no GPU and 4GB ram.
b) Handle of images for training, upload to AWS is difficult, requiere time in my current internet connection.

In order to work with the available resources I choose to compress the images from the original 160px x 320px to 64px x 64px
I use a complete turn in the track #1 in the center of the track and at a speed of 10 mph in average.
Pictures were added with specific areas of the track, the curves after the bridge and the brigde, this based on the histogram, where the counts of images with angles far from 0 were low.

![alt text][image1]

####9.- Is the model architecture documented?
The model architecture evolve from LetNet, to Nvidia, I use different parameters to test like using 3x3 convolution instead of the nvidia 5x5 convolution, since there were some issues I opt to stick to the Nvidia architecture and focus my test on the angles for left and right camera and images size, and sample size.  


![alt text][image6]

####10.- Is the creation of the training dataset and training process documented?
The training data set was generated using a mouse to control the steering, and then adding specific track areas modifying the CSV file.
Additional modeling of the data was performed by using the left and right images with the variable angle correction function, and increasing the counts of low frequency angles.

####11.- Is the car able to navigate correctly on test data?
The car is able to run the complete track 1, as shown on the video, keeping mostly the center of the track except for the curves after the bridge where the car trend to understeer.
