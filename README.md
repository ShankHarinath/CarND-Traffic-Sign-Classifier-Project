## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Link to my [project code](https://github.com/ShankHarinath/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

Dataset summary and visualizations are provided below

![Class Dist](https://raw.githubusercontent.com/ShankHarinath/CarND-Traffic-Sign-Classifier-Project/master/images/Class%20dist.png)

![Image Dist](https://raw.githubusercontent.com/ShankHarinath/CarND-Traffic-Sign-Classifier-Project/master/images/Image%20dist.png)

![Image Dist](https://raw.githubusercontent.com/ShankHarinath/CarND-Traffic-Sign-Classifier-Project/master/images/Sample%20class%20images.png)


### Design and Test a Model Architecture

#### 1. Preprocessing
---

The only preprocessing is used was normalization.
All the images are normalized to have zero mean and equal variance. This helps the model to converge soon.
An example of normalized image is also provided is below

![Preprocessed image](https://raw.githubusercontent.com/ShankHarinath/CarND-Traffic-Sign-Classifier-Project/master/images/Preprocessed%20image.png)


#### 2. Model architecture
---

###### Note: I have used Keras inside Tensorflow 1.4 to build this classifier. TF 1.4 doesn't support Cudnn 5 that comes with the AMI, I had upgrade it to Cudnn6 to make it work.

My final model consisted of the following layers:

Layer (type)         |        Description |        Output Shape         |     Param #   
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:|:---------------------------------------------:|
conv2d_1 (Conv2D)         | Filters: 64, Kernel: (1, 1), Stride: (1,1) |   (None, 32, 32, 64)     |   256 
conv2d_2 (Conv2D)          | Filters: 128, Kernel: (1, 1), Stride: (1,1) |  (None, 32, 32, 128)    |   8320      
max_pooling2d_1 (MaxPooling2 | pool_size=(2, 2) | (None, 16, 16, 128)   |    0         
batch_normalization_1 (Batch | | (None, 16, 16, 128)   |    512       
dropout_1 (Dropout)        | |  (None, 16, 16, 128)    |   0         
conv2d_3 (Conv2D)          | Filters: 256, Kernel: (3, 3), Stride: (1,1)|  (None, 14, 14, 256)    |   295168    
conv2d_4 (Conv2D)           |Filters: 512, Kernel: (3, 3), Stride: (1,1), padding=same | (None, 14, 14, 512)    |   1180160   
max_pooling2d_2 (MaxPooling2 |pool_size=(2, 2) |(None, 7, 7, 512)      |   0         
batch_normalization_2 (Batch | |(None, 7, 7, 512)      |   2048      
dropout_2 (Dropout)       | |   (None, 7, 7, 512)     |    0         
conv2d_5 (Conv2D)         | Filters: 1024, Kernel: (3, 3), Stride: (1,1)|   (None, 5, 5, 1024)    |    4719616   
conv2d_6 (Conv2D)         |Filters: 2048, Kernel: (3, 3), Stride: (1,1), padding=same |   (None, 5, 5, 2048)    |    18876416  
max_pooling2d_3 (MaxPooling2 | pool_size=(2, 2)| (None, 2, 2, 2048)   |     0         
batch_normalization_3 (Batch | | (None, 2, 2, 2048)   |     8192      
dropout_3 (Dropout)       | |   (None, 2, 2, 2048)    |    0         
flatten_1 (Flatten)       | |   (None, 8192)          |    0         
dense_1 (Dense)           | Units: 2048|   (None, 2048)          |    16779264  
dropout_4 (Dropout)       | |   (None, 2048)          |    0         
dense_2 (Dense)            | Units: 1024 |  (None, 1024)           |   2098176   
dense_3 (Dense)            | Units: 43, Activattion: softmax |  (None, 43)            |    44075     
_________________________________________________________________

Total params: 44,012,203

Trainable params: 44,006,827

Non-trainable params: 5,376

#### 3. Model parameters
---

To train the model, I used SGD optimizer with 
* Learning rate: 0.04
* Decay: 1e-6
* Momentum: 0.9
* Batch size: 60

#### 4. Model and hyperparameter tuning 
---

Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results are:
* training set accuracy of 99.99%
* validation set accuracy of 94.70% 
* test set accuracy of 94.21%

##### Models
###### Note: All the models were run with fixed hyperparameters (Batch: 60, LR: 0.01, Epocs: 20)
1. My first model was couple of Conv layers followed by a fully connected layer.
	* Accuracy: 78.535239515281918 %
2. Second model: Conv + Batch normalization + Fully connected
	* Accuracy: 84.11718717380261 %
3. Third model: Conv + Batch normalization + (Fully connected) * 3
	* Accuracy: 87.18131981666184 %
4. Fouth model: Conv + Batch normalization + (Fully connected) * 3 + Dropouts
	* Accuracy: 89.136980859797244 %
5. Fifth model: Multiple Conv + Dropout + Batch normalization + (Fully connected) * 3 + Dropouts
	* Accuracy: 92.582346117826691 % (LR: 0.01)
	* Accuracy: 93.602538505246125 % (LR: 0.03)
	* Accuracy: 94.410139223175771 % (LR: 0.04)
6. Sixth model: Multiple Conv + Dropout + Batch normalization + (Fully connected) * 3 + Dropouts + BiLSTM
	* Accuracy: 91.892324745513476 %
7. Seventh model: Resnet. I wasn't able to spend time on this model for hyperparameter optimization, but with the inital tests the results were worse than the current model.

##### Hyperparameters
###### Epocs
I have set epocs to be 40 and have implemented early stopping to prevent over fitting.

###### Batch size
Batch size is usually set to 5% of the dataset size for small datasets. Large batch size leads to lot of noise and less updates. Small batch size leads to lot of updates and takes a lot fo time to converge. Hence optimal batch size is around 60-100. I tested with batch size 60, 70 & 80. 60 & 70 leads to almost the same result. I choose 60, as it produced more consistent results.

###### Learning rate
I used LR of 0.01 to test different model architectures. Once the architecture was finalized, I tested LR = (0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08) and the best result for this dataset and model architecture is 0.04.

###### Optimizer
I have used SGD, I tried ADAM with similar hyperparameters, but the results were very bad and did not converge.

##### Model Design
Convolution layers are good at learning paterns from images. Conv layers at the lower level learns primitive patterns likt lines and arcs and the conv layers on the higher learns different ways to associate these patterns together. 

Max pooling layer is used to attend to the highest signal from the pool size and propogate only that signal higher up the network.

Batch normalization helps in faster learning and higher accuracies. BN can be used with higher learning rates. We do normalizations as part of preprocessing, with this we normalize after each conv and pooling layer.

Dropout helps generalize the model better and also doesn't tend to rely on any particular signal and activation. Dropout acts like a regularization and prevents overfitting.

Fully connected layer increases the model complexity and parameters, making it more non linear and fits better to the dataset.
 

### Test a Model on New Images

#### 1. Downloaded German traffic signs

Image 2 and 4 ahs a different background, the classifier might find it hard to classify as it might not have learnt to pick up on the actual sign. One way to prevent this is to augment data with sign placed on various backgrounds. This will ensure the model will learn to look for the sign.

Here are five German traffic signs that I found on the web:

![Bumpy Road](https://raw.githubusercontent.com/ShankHarinath/CarND-Traffic-Sign-Classifier-Project/master/images/Bumpy%20road.jpg)
![Children crossing](https://raw.githubusercontent.com/ShankHarinath/CarND-Traffic-Sign-Classifier-Project/master/images/Children%20crossing.jpg)
![No entry](https://raw.githubusercontent.com/ShankHarinath/CarND-Traffic-Sign-Classifier-Project/master/images/No%20entry.jpg)
![Road work](https://raw.githubusercontent.com/ShankHarinath/CarND-Traffic-Sign-Classifier-Project/master/images/Road%20work.jpg)
![Slippery road](https://raw.githubusercontent.com/ShankHarinath/CarND-Traffic-Sign-Classifier-Project/master/images/Slippery%20road.jpg)

The second and third image might be difficult to classify because of the background.

#### 2. Result

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Bumpy road   									| 
| Children crossing     			| Children crossing 										|
| Slippery road					| Slippery road											|
| No entry	      		| No entry					 				|
| Road work			| Dangerous curve to the right					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.21%. As the samples are less, we see 80%. Multiple runs yeild same result of 80% accuracy.

#### 3. Description
For the first image, the model is pretty sure that this is a bumpy road (probability of 0.56). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.56514317  			| Bumpy road   									| 
| 0.37446201       		| General caution							|
| 0.03678283			| Traffic signals								|
| 0.00779012 			| Dangerous curve to the right					|
| 0.00585902    		| Dangerous curve to the left      			|

For the second image, the model is not very sure that this is a Children crossing (probability of 0.38). The image can be confusing as the 32 x 32 image is very pixelated and both these labels have similar patterns. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.38344485  | Children crossing | 
| 0.29243255  |Pedestrians  | 
| 0.2490309  |Road narrows on the right  | 
| 0.04577617  |Right-of-way at the next intersection | 
| 0.01171124  			| Go straight or right  							| 

For the third image, the model is very sure that this is a Slippery road (probability of 0.86). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.86835021  			| Slippery road   									| 
| 0.08562319       		| Children crossing							|
| 0.03012757			| Ahead only				|
| 0.01021088 			| Beware of ice/snow					|
| 0.0032452    		| Dangerous curve to the right      			|

For the fourth image, the model is very sure that this is a No entry (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99021006  			| No entry   									| 
| 0.00955965       		| Turn left ahead							|
| 0.00011503			| Vehicles over 3.5 metric tons prohibited		|
| 0.00005785 			| End of no passing		|
| 0.00001929    		| Stop      			|

For the fifth image, the model is very confident that this is a Dangerous curve to the right (probability of 0.83). But the prediction is wrong. The next predicted label is the correct label and it has missed by a lot. The image can be confusing as the 32 x 32 image is very pixelated and both these labels have similar patterns. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.83033156  			| Dangerous curve to the right					| 
| 0.10155332       		| Road work					|
| 0.03623306			| Go straight or right								|
| 0.01438843 			| Right-of-way at the next intersection			|
| 0.00799581    		| Keep right      			|

#### 4. Remarks
The dataset can be augmented to have different backgrounds overlayed with the traffic signs. This will make the model learn to pick up on traffic signs rather than on the background or lighting etc.
