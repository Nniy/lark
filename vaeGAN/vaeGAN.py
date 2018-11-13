import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
import h5py
from IPython import display

import tensorflow as tf

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))


# class VAEGAN(tf.keras.Model):
