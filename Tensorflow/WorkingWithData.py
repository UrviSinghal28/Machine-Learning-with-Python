from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import urllib

import tensorflow as tf
from tensorflow import feature_column as fc

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
print(dftrain.head(),'\n') #shows the first 5 entries in the dataset
y_train = dftrain.pop('survived') #removes the survived column from dataset and stores in that variable
y_eval = dfeval.pop('survived')

print(dftrain.head(),'\n')
print(y_train,'\n')
print(dftrain.loc[0],"\n")
print(dftrain["age"],'\n')
print(dftrain.describe(),'\n')

dftrain.age.hist(bins=20).set_xlabel('age')
plt.show()

dftrain.sex.value_counts().plot(kind='barh')
plt.show()

dftrain['class'].value_counts().plot(kind='barh') #confusion about value_counts
plt.show()

pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()