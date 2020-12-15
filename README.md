# vit-tensorflow
TensorFlow wrapper of Vision Transformer from the paper "An Image Is Worth 16x16 Words" by Dosovitskiy et al.

###  Why

Inspired by Phil Wang's `vit-pytorch` [wrapper](https://github.com/lucidrains/vit-pytorch), I hoped to build something similar in TensorFlow.

###  Installation

You can install `vit-tensorflow` locally via `pip`:

```bash
pip install vit-tensorflow
```

###  Usage

As `vit-tensorflow` is a wrapper, you can use the model off-the-shelf in your pipelines.

```python
import tensorflow as tf
from vit_tensorflow import ViT

vit = ViT(
    image_size=28,
    patch_size=32,
    heads=8,
    n_classes=10,
    mlb_dim=2048,
    dropout=0.1
)

img = tf.random.uniform([28, 28], 0, 1)
logits = vit(img) # outputs a (1000, 1) vector
```

###  Citations

```
@misc{dosovitskiy2020image,
    title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author  = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year    = {2020},
    eprint  = {2010.11929},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```
@misc{vaswani2017attention,
    title   = {Attention Is All You Need},
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

###  License

[MIT](https://github.com/rish-16/vit-tensorflow/blob/main/LICENSE)