import math, cmath, numpy as np
import pandas as pd
import tensorflow as tf

i = complex(0,1)
j = complex(0,1)
k = complex(0,1)

Tupla = (i,j,k)
Data = pandas.read_csv('data/trolls.csv')

Abhorrence = complex(abhorrence_1, 1)
Hate = complex(hate_1, 1)
Rage = complex(rage_1, 1)

#Define type of tensors
Abhorrence_S = complex(A_sensation, 1)
Abhorrence_F = complex(A_feeling, 1)
Abhorrence_E = complex(A_emotion, 1)

Input_abhorrence = tf.constant([Abhorrence_S, Abhorrence_F, Abhorrence_E], dtype=tf.complex64, name= 'tensor_input')


#----Refactor in producing tensors?---


#Model
W = tf.Variable(tf.zeros[9, 3])
b = tf.Variable(tf.zeros[3])
#Create 3-d tensor for having the points information
x = tf.placeholder("float", [None, 9])


#The models comes into something like
y = tf.nn.softmax(tf.matmul(x,W) + b)
h = tf.nn.sigmoid(tf.matmul(x,W) + b)

#Defining cross entropy for fitting the model and finding better W and b
y_ = tf.placeholder("float", [None, 3])
Cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#Definition of the training step
Train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#Building the graph
Sess = tf.session()
Sess.run(tf.initialize_all_variables())

#Evaluation
Correct_prediction = tf.equal(tf.argmax(y,1), tfargmax(y_, 1))
Accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

Print sess.run(accuracy)
                
