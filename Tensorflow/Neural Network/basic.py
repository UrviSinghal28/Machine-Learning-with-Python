import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# For this tutorial we will use the MNIST Fashion Dataset. This is a dataset that is included in keras.
# This dataset includes 60,000 images for training and 10,000 images for validation/testing.
fashion_mnist = keras.datasets.fashion_mnist  # load dataset
# split into tetsing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print('\n',train_images.shape,'\n') #60,000 images made up of 28x28 pixels
print(type(train_images),'\n')
print(train_images[0,23,23],'\n') # let's have a look at one pixel
#Our pixel values are between 0 and 255, 0 being black and 255 being white. 
#This means we have a grayscale image as there are no color channels.
print(train_labels[:10],'\n')  # let's have a look at the first 10 training labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.show()