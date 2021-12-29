#Resource Link: https://www.youtube.com/watch?v=tPYj3fFJGjk&t=2043s

import tensorflow as tf
import os

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__,'\n')

#1 array/list is 1 dimension, in this rank 2 as list inside of list
rank2_tensor = tf.Variable([["test", "ok", "yes"], ["test", "yes", "ok"]], tf.string)
print(tf.rank(rank2_tensor),'\n')

#[2 3] means 2 lists with 3 elements each
print(rank2_tensor.shape,'\n')

# tf.ones() creates a shape [1,2,3] tensor full of ones ie one interior list, inside that 2 lists with 3 elements each
tensor1 = tf.ones([1,2,3])
print(tensor1,'\n')

# reshape existing data to shape [2,3,1]
tensor2 = tf.reshape(tensor1, [2,3,1])
print(tensor2,'\n')

# -1 tells the tensor to calculate the size of the dimension in that place
# this will reshape the tensor to [3,2]
tensor3 = tf.reshape(tensor2, [3, -1])
print(tensor3,'\n')

# Creating a 2D tensor
matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32) 
print(tf.rank(tensor),'\n')
print(tensor.shape,'\n')

# Selecting some different rows and columns from our tensor
three = tensor[0,2]  # selects the 3rd element from the 1st row
print(three,'\n')  # -> 3

row1 = tensor[0]  # selects the first row
print(row1,'\n')

column1 = tensor[:, 0]  # selects the first column
print(column1,'\n')

row_2_and_4 = tensor[1::2]  # selects second and fourth row
print(row_2_and_4,'\n')

column_1_in_row_2_and_3 = tensor[1:3, 0]
print(column_1_in_row_2_and_3,'\n')

# #creates a session using the default graph
# with tf.Session() as sess:
#     tensor.eval() #tensor is the name of tensor variable
