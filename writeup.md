# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

todo: number all cell bumbers 
todo: delete the old ipynb and rename the real one 
todo: update path for data 
Todo: link to code 
todo: add writeup


[//]: # (Image References)

[image1]: ./data-summary/traffic-signs.png "Traffic Signs"
[image2]: ./data-summary/training-data.png "Training Data"
[image3]: ./data-summary/validation-data.png "Validation Data"
[image4]: ./data-summary/test-data.png "Test Data"
[image5]: ./data-augmentation/bright_img.png "Brightness"
[image6]: ./data-augmentation/flip_img.png "Flipped Image"
[image7]: ./data-augmentation/org_img.png "No preprocessing"
[image8]: ./data-augmentation/rot_img.png "Rotated Image"
[image9]: ./data-augmentation/transf_img.png "Perspective Transform"
[image10]: ./data-augmentation/training-data-no-augmentation.png "No Augmentation"
[image11]: ./data-augmentation/training-data-augmentation.png "Augmentation"
[image12]: ./test-data-results/result1.png "Result: Priority road"
[image13]: ./test-data-results/result2.png "Result: Turn right ahead"
[image14]: ./test-data-results/result3.png "Result: Yield"
[image15]: ./test-data-results/result4.png "Result: Right-of-way at the next intersection"
[image16]: ./test-data-results/result5.png "Result: Stop"
[image17]: ./test-data-results/result6.png "Result: Speed limit (30km/h)"
[image18]: ./test-data-results/result7.png "Result: Wild Animals Crossing"
[image19]: ./test-data-results/result8.png "Result: No Entry"
[image20]: ./test-data-results/test-images-resized.png "Test images"


## Code Structure 

**Deliveries**: 
- Traffic_Sign_Classifier.ipynb (IPython Notebook)
- Traffic_Sign_Classifier.html (Html)

**Extra files:** 
- data: training, validation and test data 
- data-augmentation: all output images from the data augmentation
- data-summary: all output from data summary 
- results: different graphs comparing training and validation accuracies for different parameter tweaking
- test-data: the eight test images from the Internet 
- test-data-results: the top 5 predictions for each of the images in _test-data_ 
- tmp: folder where the model is stored 



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I used python, numpy and the pandas library to calculate summary statistics of the traffic
signs data set: 

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32, 32, 3)
* The number of unique classes/labels in the data set is: 43

The results can also be found in the second code cell. 
#### 2. Exploratory visualization of the dataset.

First I visualized one image of each traffic sign to get a overview of the data: 
![alt text][image1]
Then I used historgrams to visualize the distribution of the different signs in the training data, validation data and test data: 
![alt text][image2]
![alt text][image3]
![alt text][image4]

The visualizations can also be found in the third and fourth code cell. 

### Design and Test a Model Architecture

#### 1. Data Preprocessing 

I implemented several preprocessing techniques. First, I converted the images to gray scale. Even though the color can be useful information, I found that the computational gain by converting it to grayscale outweighed this. 

Then I normalized tha training data, so that the inputs were more comparable and to further improve the training efficiency.

As a last step, I experimented with other techniques to augment the training data set. I tried to even out the distribution of the training set by adding more images of the traffic signs there was less of in the training set. I did this by randomly altering the rotation, brightness, perspective transform and horizontal flipping.  

Here is an example of the preproccessing methods I experimented with: 
Original image: 
![alt text][image7]
Rotationg image 
![alt text][image8]
Adjusting  brightness: 
![Brightness][image5]
Perspective Transform: 
![alt text][image9]
Horizontal Flip: 
![alt text][image6]

After the data augmentation the training set was more evenly distributed: 
![alt text][image10]
![alt text][image11]

Surprisingly, what yielded the best validation accuracy was when I only converted to gray scale and normalized the data (~94 %, opposed to ~92 %), so I did not augment the training data augmentation. 

The data preprocessing can be found in 5th-8th code cell.

#### 2. The final model architecture 

My final model can be found in the 11th code cell - named **2.2.3 LeNet2 architecture** - and consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400    |
| RELU					|												|
| Flatten               | outputs 800                                   |    
| Fully connected		| outputs 43        							|
| Softmax				|                                               |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the LeNet-5 implementation from the classroom as a starting point. Then, I ran the method several times with different parameter values. For each iteration I tweaked either the number of epochs, batch size, optimizer or learning rate. And I ended with the choises listed in the table below. At best, this gave a validation accuracy of ~91%. This was not good engough results and I decided to work on the architecture. I tweaked the model based on the article from the article [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). I decided on this archtecture due to its good results in the same application area. I added a third concolutional layer - with no pooling layer after - and combined a flatten layer from the second and third convolutional layer as the fourth layer. Then I added dropout and a fully connected layer. I trained my model for 76 epochs, with a batchs size of 156 and a learning rate of 0.0009. The final changes improved the validation accuracy to ~94 %. 

The parameters was decided by trial and error, and can are summarized in this table; 
| Parameter     		|     Value	        					| 
|:---------------------:|:-------------------------------------:| 
| Epochs         		| 76                					| 
| Batch Size         	| 156                					| 
| Learning rate 		| 0.0009            					| 
| Droput         		| 0.95 (keep prob)  					| 
| Loss function        	| Softmax and cross entropy  			| 
| Droput         		| Adam  					            | 
| Activation function   | ReLu  					            | 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of: 1.00000
* validation set accuracy of: 0.94354
* test set accuracy of: 0.933333
My training set accuracy reached 1.00000 after a few epochs, which can indicate overfitting. I tried solving this by lowering the keep probability, but this also affected the validation accuracy. So I left the keep probability at 0.95. 

The validation and test accuracies were pretty similar, but the test set had a little lower accuracy (~1 %). Considering that the test set was three times the size, I would still argue that the model generalized well. 


### Test a Model on New Images

#### 1. Choose eight German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I tested the model on eight german traffic signs from the Internet. These are the images after being resized to 32x32:  
![alt text][image20]

The quality of most of the images remained good after the resizeing. But the icon in the seventh image was quite distorted. In addition, the seventh image was one of the traffic signs with fewer training examples. I therefore expected the model to do quite well, at least classifing seven of the images correctly. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 
The code for making predictions on my final model is located in the 16th-19th cell of the Ipython notebook. The model classified seven out of the eight images correctly - wrongly classifying the seventh image - giving it an accuracy of 0.875. This is not as high as the test set accuracy, but since I only tested the model on eight images one wrong classification impacts the results drastically. 

The model was very certain in every prediction - even when it was wrong - and guessed with nearly 100 % confidence in every prediction.. E.g. the seventh image, that should have been classified as a _Wild Animals Crossing_, was classified as a _Right-of-way at the next intersection_ with a 100 % confidence. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

Lastly, I looked at the top 5 softmax probabilities for each prediction. I visualized the reuslt for each image with bar charts: 
#### 3.1 Priority Road
The model classified the image correclty as _Priority Road_ with _100 %_ confidence
![alt text][image12]


#### 3.2 Turn Right Ahead
The model classified the image correclty as _Turn Right Ahead_ with _100 %_ confidence
![alt text][image13]


#### 3.3 Yield 
The model classified the image correclty as _Yield_ with _100 %_ confidence
![alt text][image14]

#### 3.4 Right-of-way at the next intersection
The model classified the image correclty as _Right-of-way at the next intersection_ with _97.67%_ confidence. The next guess was _Beware of ice/snow_ with _2.33 %_ confidence.
![alt text][image15] 


#### 3.5 Stop
The model classified the image correclty as _Stop_ with _97.49 %_ confidence. The next guess was _Keep Right_ with _2.51 %_ confidence.  
![alt text][image16]


#### 3.6 Speed Limit (30km/h)
The model classified the image correclty as _Speed Limit (30km/h)_ with _100 %_ confidence
![alt text][image17]


#### 3.7 Wild animals crossing
The model classified the image **wrongly** as _Right-of-way at the next intersection_ with _100 %_ confidence
![alt text][image18]

#### 3.8 No entry
The model classified the image correclty as _No entry_ with _98.77 %_ confidence. The next guess was _Keep Right_ with _1.23 %_ confidence.
![alt text][image19]