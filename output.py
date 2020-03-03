import numpy as np
import tensorflow as tf
import os
from basic_models.generator import Generator

_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz'

path_to_zip = tf.keras.utils.get_file('maps.tar.gz',
                                      origin=_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'maps/')
def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  input_image = image[:, :w, :]
  real_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

input_image, real_image = load(PATH+'train/100.jpg')
generator = Generator()

prediction = generator(input_image, training=False)   
    
title = ['Input Image', 'Ground Truth', 'Predicted Image']
for i in range(5):
    display_list = [input_image[i], real_image[i], prediction[i]]
    for j in range(3):
        plt.subplot(6, 3, i*3+j+1)
        plt.title(title[j])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[j] * 0.5 + 0.5)
        plt.axis('off')
plt.show()

