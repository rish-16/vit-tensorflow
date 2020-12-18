import numpy as np
import tensorflow as tf
from vit_tensorflow import MultiHeadAttention

x = tf.random.uniform([28, 28], 0, 1)
MHA = MultiHeadAttention(784, 4)
out = MHA(x)

print (x.shape)
print (out.shape)