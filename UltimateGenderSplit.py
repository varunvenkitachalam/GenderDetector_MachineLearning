
"""
Created on Sat Oct 20 16:01:38 2018

@author: Varun
"""

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import shutil

#reads in csv file
df = pd.read_csv("data/train_target.csv")
df1 = df['Id']
testLabels = []
validateLabels = []
jpg = []
i = 0
j = 0
#keeps track of number of files
trainMale = 0
trainFemale = 0
validateMale = 0
validateFemale = 0

leastGender = 0

#separates data into two Male/ Female Directories
while i < 28360:
    print(df1.iloc[i])
    imageName = ('data/train/' + df1.iloc[i])
    testTarget = df.iloc[i][1]
    if (testTarget == 1):
        trainMale = trainMale +1
        shutil.copy(imageName, "data/Split/Male/Male" + str (trainMale) + ".jpg" )
    else:
        trainFemale = trainFemale +1
        shutil.copy(imageName, "data/Split/Female/Female" + str (trainFemale) + ".jpg" )
    i = i+1

if(trainMale > trainFemale):
    leastGender = trainFemale
else:
    leastGender = trainMale
    
print(leastGender)
k = 1
while(k<leastGender +1):
    genderSwitch = 0
    imageName = ("data/Split/")
    if(k < 1 +leastGender - 2000):
        if(genderSwitch == 0):
            shutil.move(imageName+"Male/Male"+ str (k) + ".jpg", "data/Ultimate_train/Male/Male" + str (k) + ".jpg" )
            genderSwitch = 1
        if(genderSwitch == 1):
            shutil.move(imageName+"Female/Female"+ str (k) + ".jpg", "data/Ultimate_train/Female/Female" + str (k) + ".jpg" )
            genderSwitch = 0
    else:
        if(genderSwitch == 0):
            shutil.move(imageName+"Male/Male"+ str (k) + ".jpg", "data/Ultimate_validation/Male/Male" + str (k) + ".jpg" )
            genderSwitch = 1
        if(genderSwitch == 1):
            shutil.move(imageName+"Female/Female"+ str (k) + ".jpg", "data/Ultimate_validation/Female/Female" + str (k) + ".jpg" )
            genderSwitch = 0
    k = k+1

FemCounter = 1;
while(k<(leastGender*2) +1):
    imageName = ("data/Split/")
    if(k <  (leastGender*2) - 2000 +1):
        shutil.move(imageName+"Male/Male"+ str (k) + ".jpg", "data/Ultimate_train/Male/Male" + str (k) + ".jpg" )
        shutil.copy("data/training_set1/Female/Female" + str (FemCounter) + ".jpg", "data/Ultimate_train/Female/Female" + str (k) + ".jpg" )
    else:
        shutil.move(imageName+"Male/Male"+ str (k) + ".jpg", "data/Ultimate_validation/Male/Male" + str (k) + ".jpg" )
        shutil.copy("data/validation_set1/Female/Female" + str (FemCounter) + ".jpg", "data/Ultimate_validation/Female/Female" + str (k) + ".jpg" )
    k = k+1
    FemCounter = FemCounter +1

FemCounter = 1;
while(k <  (leastGender*3) - 2000 +1):
    
    shutil.move(imageName+"Male/Male"+ str (k) + ".jpg", "data/Ultimate_train/Male/Male" + str (k) + ".jpg" )
    shutil.copy("data/training_set1/Female/Female" + str (FemCounter) + ".jpg", "data/Ultimate_train/Female/Female" + str (k) + ".jpg" )
    k = k+1
    FemCounter = FemCounter +1




