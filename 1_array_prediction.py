#!/usr/bin/env python3
"""
example, if you were writing code for a function like this,
in traditional programming, you already know the 'rules' —

float hw_function(float x){
    float y = (2 * x) - 1;
    return y;
}

using a neural network (or any other Machine learning)
requires you to input both data and answers to generate rules

below is a super simple NN to demonstrate this idea.
"""

# importing library
import numpy as np
import tensorflow as tf
from tensorflow import keras

# define a model with 1 Dense layer, 1 neuron (units) and 1 input shape
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# compile the model with appropriate optimizer function and loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# provide data in array format
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# train the NN for 500 epochs
model.fit(xs, ys, epochs=500)

# display the result
print(model.predict([10.0]))