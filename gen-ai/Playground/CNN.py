import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np 

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train/255.0, x_test/255.0
print(x_train, x_test)