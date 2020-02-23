# GenderDetector_MachineLearning
Created a Binary Gender Detector using Convolutional Neural Networks to successfully predict the gender of a person based off of a picture of their face. Obtained a log loss of 0.15932

UltimateGenderSplit.py: This reads in the large dataset of images with the different faces, and attempts to properly split the data with the same amount of male to female images. This way the model will prevent overfitting by having a large amount of one datatype.  

 CrazyAlteredGenderCombo.py: This file uses convolutional neural networks to properly create, train, validate, and test our model based off of the data that is fed inside. We adjust our learning rate, dropout level, batch size, and early stop mechanism in this file. Finally, we use our newly created model to predict the images from our test dataset.
 
 Crazypredictions32FineTuned.csv: This file is our final predictions from our model with 1 representing male and 0 representing female. This is the 32nd iteration of this model and has fairely beneficial results, showing us the probability of the model predicting whether the image is male or female. 
 
 Project Analysis: This file wraps up the project by focusing on the technical aspects of how the model was created including the training, validation, and testing portions. Also explains why the log loss is relatively low through proper fine tuning and transfer learning. 
