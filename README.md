# emotion detection using FER2013 dataset

Implementation 1:

	- csv2image.py : to extract training, validation and test images from fer2013 csv file
	- cnn_models.py : contains models' architecture
	- emotion_detection.py : to train the model and save in .h5 format and analyse accuracy and loss respect to epoch
	- real_prediction.py : to determine emotion of the person using openCV and saved models

	- accuracy folder contains accuracy of the model respect to epoch
	- loss folder contains validation loss of the model respect to epoch 

	result: 
		- Best model is one layer convolution network with around 45% acuuracy
		- Models used in this project were not up to the mark
		- Training the complex model using available dataset was not a good choice

	Problems:
		- Very slow prediction due to low computing power

NOTE: Only used 6 emotions for prediction due to less number of images available for disgusting emotion

		
Implementation 2:

	- Used batch size of 64 to train the model (keeping the size small helps to reduce the training loss)
	- Used 90% of the data for training and 10% for validation
	
	result:
		- All the models performed better compare to the 1st implementation
		- mini_XCEPTION resulted ~77% training accuracy and ~57% validation accuracy
	
NOTE: Did not train all the models due to limited resources available on FloydHub account


FloydHub:

	- To avoid the problem of slow prediction
	- Trained models on GPU
	  Usage: 
		for GPU, floyd run --gpu --data path:filename --follow 'python new_emotion_detection.py'
		for CPU, floyd run --cpu --data path:filename --follow 'python new_emotion_detection.py'

	
Download:

	- Download the fer2013.csv file at https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data 


Reference:
 
	https://github.com/bartchr808/EmotionPredictingCNN/blob/master/README.md
	https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data