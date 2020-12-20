import numpy as np
import sonnet as snt
import tensorflow as tf

class Residual(snt.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def __call__(self, x):
        return self.fn(x) + x
        
class PreNorm(snt.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = snt.LayerNorm(dim, create_scale=False, create_offset=False)
        self.fn = fn
        
    def __call__(self, x):
        return self.fn(self.norm(x))
        
class MLP(snt.Module):
    def __init__(self, dim, h_dim, dropout=0.1):
        super().__init__()
        self.l1 = snt.Linear(h_dim)
        self.l3 = snt.Dropout(dropout)
        self.l4 = snt.Linear(dim)
        self.l5 = snt.Dropout(dropout)
        
    def __call__(self, x):
        x = tf.nn.gelu(self.l1(x))
        x = self.l3(x)
        x = self.l4(x)
        out = self.l5(x)
        
        return out        

class MultiHeadAttention(snt.Module):
    def __init__(self, dim, n_heads, dropout=0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        
        assert dim % n_heads == 0
        
        self.W_q = snt.Linear(dim, with_bias=False)
        self.W_k = snt.Linear(dim, with_bias=False)
        self.W_v = snt.Linear(dim, with_bias=False)
        
        self.W_h = snt.Linear(dim, with_bias=False)

    def scaled_dot_product_attn(self, Q, K, V):
        Q = Q / np.sqrt(self.dim)    
        scores = tf.matmul(Q, tf.transpose(K))
        A = tf.nn.softmax(scores)
        H = tf.matmul(A, V)
        
        return H, A
        
    def __call__(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        H_cat, A = self.scaled_dot_product_attn(Q, K, V)
        
        return self.W_h(H_cat)
        
class EncoderBlock(snt.Module):
    def __init__(self, dim, n_heads, conv_hidden, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        
        self.l1 = MultiHeadAttention(dim, n_heads=n_heads, dropout=dropout)
        self.l2 = snt.LayerNorm(dim, create_scale=False, create_offset=False)
        self.l3 = snt.LayerNorm(dim, create_scale=False, create_offset=False)
        
    def __call__(self, x):
        attn_op = self.l1(x)        
        out1 = self.l2(attn_op)
        out2 = self.l3(out1)
        
        return out2

class TransformerEncoder(snt.Module):
    def __init__(self, dim, depth, n_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = []
        dim = dim
        depth = depth
        n_heads = n_heads
        mlp_dim = mlp_dim
        dropout = dropout
        
        for _ in range(depth):
            self.layers.append(EncoderBlock(dim, n_heads, mlp_dim, dropout))
            
    def __call__(self, x):
        for i in range(self.depth):
            x = self.layers[i](x)
            
        return x

class ViT(snt.Module):
    def __init__(self, 
                 img_size, 
                 patch_size, 
                 n_classes, 
                 n_heads=8, 
                 mlp_dim=2048):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by Patch size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_classes = n_classes
        
    def __call__(self, img):
        pass