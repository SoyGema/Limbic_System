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

#3.--------------- LOAD IMAGES AND PROCESS THEM------------------------
 
# Move through the directory and get the image paths. The Y lables are encoded in the file name
# like Vigilance_203.jpg 
  
def get_filepaths(path):
    data = []
    for r,d,f in os.walk(path):
        for file in f:
            if '.jpg' or '.jpge' in file:
                full_name = os.path.join(r,file)
                label = file.split('_')[1].split('.')[0]
                data.append([full_name, y_labels.get(label)])

    return data 

def process_images(files):
    n = len(files)
    for i in range(n):
        # Load image and use scharr to get edges
        image = io.imread(files[i][0],as_grey=True)
        image = scharr(img)
        
        # Scale the intensity to whiten the edges
        p95 = np.percentile(img, (95))
        image = exposure.rescale_intensity(image, in_range=(0, p95))
        
    return image


def get_filepaths(path):
    data = []
    
    # Move through the directory and get the image paths. The Y lables are encoded in the file name
    # like homer_203.jpg
    for r,d,f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                full_name = os.path.join(r,file)
                label = file.split('_')[1].split('.')[0]
                data.append([full_name, y_labels.get(label)])                   
    return data
                
def process_images(files, x_size, y_size):
    # Flip each image upside down so we double the array to accomidate
    arr = np.zeros(shape=(len(files)*2, x_size , y_size))
    n = len(files)
    for i in range(n):
        # Load image and use scharr to get edges
        image = io.imread(files[i][0],as_grey=True)
        image = scharr(img)
        
        # Scale the intensity to whiten the edges
        p95 = np.percentile(img, (95))
        image = exposure.rescale_intensity(img, in_range=(0, p95))
        
        # Add image to the collection. Rotate it 180 and add another copy.
        arr[i] = img
        image = rotate(img, angle=180)
        arr[i+n] = img
        
    return arr

def get_image_data():
    # Load all images in the directory and generate feature & label data
    base_path = 'C:/path_where_the_images_are'
    files = get_filepaths(base_path)
    # Standard size for mobile pics  
    x_data = process_images(files, 600, 749)
    y_data = np.array(list([y for (x,y) in files])    
    return shuffle(x_data, y_data)

X, y = get_image_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

 
