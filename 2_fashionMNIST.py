#!usr/bin/env python3

"""
In the previous exercise you saw how to create a neural network
 that figured out the problem you were trying to solve.
  This gave an explicit example of learned behavior.
   Of course, in that instance, it was a bit of overkill because 
   it would have been easier to write the function Y=2x-1 directly,
    instead of bothering with using Machine Learning to learn
     the relationship between X and Y for a fixed set of values
     , and extending that for all values.

But what about a scenario where writing rules like that is much
 more difficult -- for example a computer vision problem?
  Let's take a look at a scenario where we can recognize different
   items of clothing, trained from a dataset containing 10 different types.
"""

# import package
# import tensorflow as tf
# print(tf.__version__)
# import numpy as np
# import matplotlib.pyplot as plt

# # accessing dataset directly from tf.keras datasets API
# mnist = tf.keras.datasets.fashion_mnist

# # loading into train and test set
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# # The number of characters per line for the purpose of inserting line breaks (default 75).
# np.set_printoptions(linewidth=200)

# # displaying first image, printing its labels and image pixel in array form
# # you can change the index to view other images
# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])

# # normalization
# training_images = training_images/255.0
# test_images = test_images/255.0

# # model configuration
# # Sequential = defines a sequence of layers in the NN
# # Flatten = turns t into a 1D array
# # Dense = add a layer of interconnected neurons
# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# # model building
# model.compile(loss = 'sparse_categorical_crossentropy',
#                 optimizer = tf.optimizers.Adam(),
#                 metrics=['accuracy'])
# model.fit(training_images, training_labels, epochs=5)

# # model evaluation
# model.evaluate(test_images, test_labels)

# # perform prediction - Q: is the prediction accurate?
# classifications = model.predict(test_images)
# print(classifications[0])
# print(test_labels[0])

# Optional Exercise 1
# Please uncomment only these parts and comment out the rest
# in order to run
# Q: does increase in neurons affect model accuracy and training time?
# import tensorflow as tf

# mnist = tf.keras.datasets.mnist

# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# training_images = training_images/255.0
# test_images = test_images/255.0

# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(1024, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# model.compile(optimizer='adam',
#                loss='sparse_categorical_crossentropy')

# model.fit(training_images, training_labels, epochs=5)

# model.evaluate(test_images, test_labels)

# classifications = model.predict(test_images)

# print(classifications[0])
# print(test_labels[0])

# Optional Exercise 2
# Q: Does adding an additional layer affect the training time and model accuracy?

# import tensorflow as tf
# print(tf.__version__)

# mnist = tf.keras.datasets.mnist

# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# training_images = training_images/255.0
# test_images = test_images/255.0

# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(512, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(256, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# model.fit(training_images, training_labels)

# model.evaluate(test_images, test_labels)

# classifications = model.predict(test_images)

# print(classifications[0])
# print(test_labels[0])

# Optional Exercise 3 and 4
# Investigate the effect of number of training epochs and normalization

import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# comment the lines below to test for normalization
training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=30)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])