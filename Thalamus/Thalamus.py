#CNN Structure as a Thalamus
#InceptionV3 + #CNN_new structure


# 1. -------------- IMPORT WORK STATEMENTS --------------------

import tensorflow as tf
import numpy as np
import math, os, random

from skimage import io, exposure
from skimage.transform import rotate
from skimage.filters import scharr

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# 2. ------------- CREATE LABELS FOR EMOTION CLASSIFIER ------------
#In this case, we create two dictionaries, one for classification and another one for inference

# Y for the classifier 
y_clas = {
  'Annoyance' : 0,
  'Hate' : 1,
  'Rage' : 2,
  'Contempt' : 3,
  'Loathing' : 4,
  'Disgust' : 5,
  'Boredom' : 6
  }
  
# Y for inference
y_inf = {
  'Vigilance' : 0,
  'Anticipation' : 1,
  'Interest' : 2,
  'Grief' : 3,
  'Sadness' : 4,
  'Pensiveness' : 5
  }

#3.--------------- LOAD IMAGES ------------------------
 
 def get_filepaths(path):
    data = []
    
    # Move through the directory and get the image paths. The Y lables are encoded in the file name
    # like Vigilance_203.jpg
    for r,d,f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                full_name = os.path.join(r,file)
                label = file.split('_')[1].split('.')[0]
                data.append([full_name, y_labels.get(label)])
            if '.jpge' in file:
                full_name = os.path.join(r,file)
    return data 

