# emotion detection using FER2013 dataset

- csv2image.py : to extract training, validation and test images from fer2013 csv file
- cnn_models.py : contains models' architecture
- emotion_detection.py : to train the model and save in .h5 format and analyse accuracy and loss respect to epoch
- real_prediction.py : to determine emotion of the person using openCV and saved models

- accuracy folder contains accuracy of the model respect to epoch
- loss folder contains validation loss of the model respect to epoch 

Conclusion: 
- Best model is one layer convolution network with around 45% acuuracy
- Models used in this project were not up to the mark
- Training the complex model using available dataset was not a good choice

Problems:
- Very slow prediction due to low computing power 

Download:
- Download the fer2013.csv file at https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data 

Reference:
 
	https://github.com/bartchr808/EmotionPredictingCNN/blob/master/README.md
	https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data