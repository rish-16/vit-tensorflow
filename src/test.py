import numpy as np
import tensorflow as tf
from vit_tensorflow import MultiHeadAttention
import matplotlib.pyplot as plt

(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
img = xtrain[0].reshape([784, 1]) / 255

MHA = MultiHeadAttention(28, 4)
out = MHA(img)

plt.figure()
plt.subplot(121)
plt.imshow(img.reshape([28, 28]), cmap="gray")

plt.subplot(122)
plt.imshow(out, cmap="gray")

plt.show()