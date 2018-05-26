# **Traffic Sign Recognition**

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/Visualization.png "Visualization"
[image2]: ./writeup_images/lenet.png "LeNet Architecture"
[image3]: ./writeup_images/RoadWork.png  "Traffic Sign 1"
[image4]: ./writeup_images/SpeedLimit.png "Traffic Sign 2"
[image5]: ./writeup_images/Stop.png "Traffic Sign 3"
[image6]: ./writeup_images/TurnRight.png "Traffic Sign 4"
[image7]: ./writeup_images/WildAnimals.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/wwha/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing that the example for each class
are not distributed evenly. There are more examples of some classes over others.


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to randomize the train dataset so that the learning could use random data and the results reflect
the common learning.

Then, I converted the images to grayscale because it reduce the effect of the color and generates
better results.

As a last step, I normalized the image data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|				outputs 28x28x6								|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16	|
| RELU                | outputs 10x10x16      |
| Max pooling         | 2x2 stride, outputs 5x5x16      |
| Flatten          | outputs 400        |
| Fully connected		| inputs 400, outputs  120   |
| RELU        | outputs 120     |
| Fully connected   | inputs 120, outputs 84     |
| RELU        | outputs 84       |
| Fully connected   | inputs 120, output 43      |




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the batch size of 128, number of epochs of 40, learning rate of 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.951
* test set accuracy of 0.929

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
The architecture of the LeNet is chosen. Here is the architecture map.
![alt text][image2]
* Why did you believe it would be relevant to the traffic sign application?
After running the architecture, it did generate a validation accuracy of around 0.8, which is good starting point of the project.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The validation accuracy started from 0.829. After the 34th epoch, the accuracy stayed around 0.951, which was good accuracy to tell that the model worked. And the final accuracy of the training and test also showed model worked.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5]
![alt text][image6] ![alt text][image7]

The first image might be difficult to classify because the Truck besides the sign has a bright color.
The second image might be difficult to classify because the sign is in the corner of the image.
The third image might be difficult to classify because of the background trees.
The fourth image might be difficult to classify because of the other signs in the corner.
The fifth image might be difficult to classify because the edge of the sign is not clear.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road Work     		| Go Straight or Left   						|
| Speed Limit (60km/h)  | Speed Limit (60km/h) 							|
| Stop					| Stop										    |
| Turn Right Ahead	    | Turn Right Ahead					 			|
| Wild Animals Crossing	| Wild Animals Crossing      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

For the first image, the model does provide bad prediction of Road Work, which is not in the top five soft max probabilities below.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .997         			| Go Straight or Left   						|
| .002    				| General Caution 								|
| .001					| Speed Limit (30km/h)							|
| 0	      			    | Traffic Signals					 			|
| 0				        | Speed Limit (20km/h)      					|


For the second image, the model provide good prediction of Speed Limit (60km/h), which has the maximum probability of 0.998. The top five soft max probability is listed below

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .998         			| Speed Limit (60km/h)   						|
| .001    				| Keep Right									|
| .001					| Speed Limit (30km/h)					        |
| 0	      			    | End of Speed Limit (80km/h)					|
| 0				        | Yield      							        |

For the third image, the model provide good prediction of Stop, which has the maximum probability of 1. The top five soft max probability is listed below

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Stop   									    |
| 0   				    | Turn Right Ahead 								|
| 0					    | No Entry										|
| 0	      			    | Yield					 				        |
| 0				        | Keep Right      							    |

For the fourth image, the model provide good prediction of Turn Right Ahead, which has the maximum probability of 0.999. The top five soft max probability is listed below

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .999         			| Turn Right Ahead   							|
| .001    				| Ahead Only 									|
| 0					    | Yield											|
| 0	      			    | Stop					 				        |
| 0				        | Keep Right      						        |

For the second image, the model provide good prediction of Wild Animals Crossing, which has the maximum probability of 1. The top five soft max probability is listed below

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Wild Animals Crossing   						|
| 0    				    | Double Curve 									|
| 0					    | Right-of-Way at the Next Intersection			|
| 0	      			    | Speed Limit (20km/h)				 			|
| 0				        | Bicycles Crossing     						|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
