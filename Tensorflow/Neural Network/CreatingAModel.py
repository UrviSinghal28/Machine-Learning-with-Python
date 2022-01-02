import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset
# split into tetsing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#The last step before creating our model is to preprocess our data. 
# This simply means applying some prior transformations to our data before feeding it the model. 
# In this case we will simply scale all our greyscale pixel values (0-255) to be between 0 and 1. 
# We can do this by dividing each value in the training and testing sets by 255.0. 
# We do this because smaller values will make it easier for the model to process our values.
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])
# Layer 1: This is our input layer and it will conist of 784 neurons. 
# We use the flatten layer with an input shape of (28,28) to denote that our input should come in in that shape. 
# The flatten means that our layer will reshape the shape (28,28) array into a 
# vector of 784 neurons so that each pixel will be associated with one neuron.

# Layer 2: This is our first and only hidden layer. The dense denotes that this layer will be fully connected and 
# each neuron from the previous layer connects to each neuron of this layer. 
# It has 128 neurons and uses the rectify linear unit activation function.

# Layer 3: This is our output later and is also a dense layer. 
# It has 10 neurons that we will look at to determine our models output. 
# Each neuron represnts the probabillity of a given image being one of the 10 different classes. 
# The activation function softmax is used on this layer to calculate a probabillity distribution for each class. 
# This means the value of any neuron in this layer will be between 0 and 1, 
# where 1 represents a high probabillity of the image being that class.

