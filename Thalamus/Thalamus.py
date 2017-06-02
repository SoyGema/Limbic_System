#CNN Structure as a Thalamus
#InceptionV3 + #CNN_new structure
# The following is adopted from the code provided by Google's Martin Gorner and John Becina article about Tensorboard

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

 tf.logging.set_verbosity(tf.logging.INFO)

# ********************************************************************
# Define some functions first for use in the model (keeps things neat)
# ********************************************************************

# Function feeds data into our model
## When trying the model we can use another batch_size that will affect performance                       
def train_data_input_fn():
    return tf.train.shuffle_batch([tf.constant(X_train, dtype=tf.float32), tf.constant(y_train, dtype=tf.int32)],
                                  batch_size=100, capacity=1100, min_after_dequeue=1000, enqueue_many=True)


# Use cross entropy to define our loss function which we are trying to minimize
## Here remember that the significant issue is to reduce the loss function 
def conv_model_loss(Ylogits, Y_, mode):
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        return tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,4), Ylogits))
    else:
        return None


# Define which optimizer we are using to minimize our loss. This one incorporates learning decay
## When trying the model we can use Adam, Adagrad or GradientDescentOptimizer 
def conv_model_train_op(loss, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.contrib.layers.optimize_loss(
            loss, 
            tf.train.get_global_step(), 
            learning_rate=0.003, 
            optimizer="Adam",
            learning_rate_decay_fn=lambda lr, step: 0.0001 + tf.train.exponential_decay(lr, step, -3000, math.e)
        )
    else:
        return None


# Track accuracy, precision, recall metrics during our training and eval steps
def conv_model_eval_metrics(classes, Y_, mode):
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        return {
            'accuracy': tf.metrics.accuracy(classes, Y_),
            'precision': tf.metrics.precision(classes, Y_),
            'recall': tf.metrics.recall(classes, Y_),
        }
    else:
        return None

    # Define our actual model
def cnn_model_fn(features, labels, mode):
    
    
    input_layer = tf.reshape(features, [-1, 600, 749, 1])
    tf.summary.image('input', input_layer, 10) #Logs 10 sample images to view in TensorBoard
    
    # Convolutional Layer #1 - Out 300x375 image because stride 2
##Change stride and it will change the output 
    with tf.name_scope('cnn_layer1_8x8x8-s2'):
        conv1 = tf.layers.conv2d(
            strides=(2, 2),
            inputs=input_layer,
            filters=8,
            kernel_size=[8, 8],
            padding="same",
            use_bias=False,
            activation=None)

        bn1 = tf.layers.batch_normalization(conv1, training=mode == tf.estimator.ModeKeys.TRAIN)
        re1 = tf.nn.relu(bn1)
        tf.summary.histogram('weights', conv1) #These add histograms to view in TensorBoard
        tf.summary.histogram('bias', bn1)
        tf.summary.histogram('activations', re1)
    
    # Convolutional Layer #2 - Out 150 x 162 image because stride 2
    with tf.name_scope('cnn_layer2_6x6x16-s2'):
        conv2 = tf.layers.conv2d(
            strides=(2, 2),
            inputs=re1,
            filters=16,
            kernel_size=[6, 6],
            padding="same",
            use_bias=False,
            activation=None)

        bn2 = tf.layers.batch_normalization(conv2, training=mode == tf.estimator.ModeKeys.TRAIN)
        re2 = tf.nn.relu(bn2)
        tf.summary.histogram('weights', conv2)
        tf.summary.histogram('bias', bn2)
        tf.summary.histogram('activations', re2)

    # Convolutional Layer #3 - Out 75 x 80 image because stride 2
    with tf.name_scope('cnn_layer3_3x3x48-s2'):
        conv3 = tf.layers.conv2d(
            strides=(2, 2),
            inputs=re2,
            filters=36,
            kernel_size=[3, 3],
            padding="same",
            use_bias=False,
            activation=None)  

        bn3 = tf.layers.batch_normalization(conv3, training=mode == tf.estimator.ModeKeys.TRAIN)
        re3 = tf.nn.relu(bn3)
        tf.summary.histogram('weights', conv3)
        tf.summary.histogram('bias', bn3)
        tf.summary.histogram('activations', re3)
    
    # Flatten so we can use the output ## 36 number is because of the filters 
    re3_flat = tf.reshape(re3, [-1, 75 * 80 * 36])
    
    # Dense layer - 2048 hidden nodes with a dropout rate of 30%
    with tf.name_scope('dense_layer4_10800x2048'):
        dense = tf.layers.dense(inputs=re3_flat, units=2048, activation=None, use_bias=False)
        bn_dense = tf.layers.batch_normalization(dense, training=mode == tf.estimator.ModeKeys.TRAIN)
        re_dense = tf.nn.relu(bn_dense)


        tf.summary.histogram('weights', dense)
        tf.summary.histogram('bias', bn_dense)
        tf.summary.histogram('activations', re_dense)
    
        dropout4 = tf.layers.dropout(
          inputs=re_dense, rate=0.30, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Dense layer - 1024 hidden nodes with a dropout rate of 30%
    with tf.name_scope('dense_layer5_2048x1024'):
        dense5 = tf.layers.dense(inputs=dropout4, units=1024, activation=None, use_bias=False)
        bn_dense5 = tf.layers.batch_normalization(dense5, training=mode == tf.estimator.ModeKeys.TRAIN)
        re_dense5 = tf.nn.relu(bn_dense5)


        tf.summary.histogram('weights', dense5)
        tf.summary.histogram('bias', bn_dense5)
        tf.summary.histogram('activations', re_dense5)
    
        dropout5 = tf.layers.dropout(
          inputs=re_dense5, rate=0.30, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Final layer of 4 nodes which we will apply softmax to for prediction
    with tf.name_scope('output_layer5'):
        logits = tf.layers.dense(inputs=dropout5, units=4)
        tf.summary.histogram('weights', logits)
    
    predict = tf.nn.softmax(logits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)
    
    # Populate these using our helpful functions above
    loss = conv_model_loss(logits, labels, mode)
    train_op = conv_model_train_op(loss, mode)
    eval_metrics = conv_model_eval_metrics(classes, labels, mode)
    
    with tf.variable_scope("performance_metrics"):
        tf.summary.scalar('accuracy', eval_metrics['accuracy'][1])
        tf.summary.scalar('precision', eval_metrics['precision'][1])
        tf.summary.scalar('recall', eval_metrics['recall'][1])

    # This is a required return format
    return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions={"predictions": predict, "classes": classes}, # name these fields as you like
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metrics)

# Build our clasifier
simp_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/thalamus")

# This is the actual training step. You can interrupt and resume it as needed since it's checkpointed
simp_classifier.train(input_fn=train_data_input_fn, steps=5000)
