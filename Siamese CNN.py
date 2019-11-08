# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:58:33 2019
Student Name & No. : Jianwei Tang N10057862, Yongrui Pan N10296255
                    
@author: User
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import random
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, MaxPooling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
from keras.models import Model, Sequential

from tensorflow.keras import regularizers

import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# normalize the data between 0–1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# The position of ["top", "trouser", "pullover", "coat", "sandal", "ankle boot"] are [0,1,2,4,5,9] 
# split labels ["top", "trouser", "pullover", "coat", "sandal", "ankle boot"] to train set
digit_indices = [np.where(y_train == i)[0] for i in {0,1,2,4,5,9}]
digit_indices = np.array(digit_indices)

# length of each column
n = min([len(digit_indices[d]) for d in range(6)])

# The position of ["dress", "sneaker", "bag", "shirt"] are [3,6,7,8]
# Keep 80% of the images with labels ["top", "trouser", "pullover", "coat", "sandal", "ankleboot"] for training (and 20% for testing)
train_set_shape = n * 0.8
test_set_shape = n * 0.2
y_train_new = digit_indices[:, :int(train_set_shape)]
y_test_new = digit_indices[:, int(train_set_shape):]

# Keep 100% of the images with labels in ["dress", "sneaker", "bag", "shirt"] for testing
digit_indices_t = [np.where(y_train == i)[0] for i in {3,6,7,8}]
y_test_new_2 = np.array(digit_indices_t)

# Keep 100% of the images with labels ["top", "trouser", "pullover", "coat","sandal", "ankle boot"] union ["dress", "sneaker", "bag", "shirt"] for testing
digit_indices_3 = [np.where(y_train == i)[0] for i in range(10)]
y_test_new_3 = np.array(digit_indices_3)

def create_pairs(x, digit_indices):
    '''
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    # labels are 1 or 0 identify whether the pair is positive or negative
    labels = []
    
    class_num = digit_indices.shape[0]
    for d in range(class_num):
        for i in range(int(digit_indices.shape[1])-1):
            # use images from the same class to create positive pairs
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            # use random number to find images from another class to create negative pairs
            inc = random.randrange(1, class_num)
            dn = (d + inc) % class_num
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            # add two labels which the first one is positive class and the second is negative.
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

# For training, images are from 80% of the images with labels ["top", "trouser", "pullover", "coat", "sandal", "ankleboot"]
tr_pairs, tr_y = create_pairs(x_train, y_train_new)
# Reshape for the convolutional neural network, same for the test sets below.
tr_pairs = tr_pairs.reshape(tr_pairs.shape[0], 2, 28, 28, 1)

# For testing, images are from the rest 20% of the images with labels ["top", "trouser", "pullover", "coat", "sandal", "ankleboot"]
te_pairs_1, te_y_1 = create_pairs(x_train, y_test_new)
te_pairs_1 = te_pairs_1.reshape(te_pairs_1.shape[0], 2, 28, 28, 1)

# For testing, images are from 100% of the images with labels in ["dress", "sneaker", "bag", "shirt"]
te_pairs_2, te_y_2 = create_pairs(x_train, y_test_new_2)
te_pairs_2 = te_pairs_2.reshape(te_pairs_2.shape[0], 2, 28, 28, 1)

# For testing, images are from 100% of the whole dataset
te_pairs_3, te_y_3 = create_pairs(x_train, y_test_new_3)
te_pairs_3 = te_pairs_3.reshape(te_pairs_3.shape[0], 2, 28, 28, 1)

def create_base_network(input_shape):
    '''
    Base network to be shared.
    '''
    input = Input(shape=input_shape)
    x = Conv2D(32, (7, 7), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01),  
              bias_regularizer=regularizers.l1(0.01))(input)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3 ,3), activation='relu', kernel_regularizer=regularizers.l2(0.01),  
              bias_regularizer=regularizers.l1(0.01))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),  
              bias_regularizer=regularizers.l1(0.01))(x)

    return Model(input, x)

# input shape
input_shape = (28,28,1)

base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

epochs = 10
# reference from keras example "https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py"
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# add a lambda layer
distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs_1[:, 0], te_pairs_1[:, 1]], te_y_1))

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs_1[:, 0], te_pairs_1[:, 1]])
te_acc = compute_accuracy(te_y_1, y_pred)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
    
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print('-------------------------------------------------------------------------------')
print('* Test set 1')
print('* 20% of the images with labels ["top", "trouser", "pullover", "coat", "sandal", "ankleboot"]')
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

# predict test set 2
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs_3[:, 0], te_pairs_3[:, 1]])
te_acc = compute_accuracy(te_y_3, y_pred)

print('-------------------------------------------------------------------------------')
print('* Test set 2')
print('* 100% of the images with labels in ["dress", "sneaker", "bag", "shirt"]')
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

# predict test set 3
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs_2[:, 0], te_pairs_2[:, 1]])
te_acc = compute_accuracy(te_y_2, y_pred)

print('-------------------------------------------------------------------------------')
print('* Test set 3')
print('* Images form the whole data set')
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))