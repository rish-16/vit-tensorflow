import numpy as np
import sonnet as snt
import tensorflow as tf

class MultiHeadAttention(snt.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = 1 / np.sqrt(dim)
        self.heads = heads
        
    def __call__(self, x):
        pass
        
class Transformer(snt.Module):
    def __init__(self):
        super().__init__()

class ViT(snt.Module):
    def __init__(self, 
                 img_size, 
                 patch_size, 
                 n_classes, 
                 n_heads=8, 
                 mlp_dim=2048):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_classes = n_classes