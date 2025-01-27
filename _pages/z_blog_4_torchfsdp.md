---
permalink: /blog_4_torchfsdp.html
title: "blog_4_torchfsdp"
author_profile: true
date: 2024-01-25
layout: blog
toc: true
---

# LiteLlama: A Lightweight Transformer for Efficient Language Modeling

- Xiaotian Han
- 01/25/2024

## TL;DR
We propose LiteLlama, a lightweight transformer model for efficient language modeling. LiteLlama is designed to be more efficient than the original transformer model while maintaining competitive performance. We achieve this by reducing the model size and the number of parameters. We also introduce a new position embedding method that is more efficient than the original method. We evaluate LiteLlama on the WikiText-103 dataset and show that it outperforms the original transformer model in terms of speed and memory usage while maintaining competitive performance.

### Introduction
In this blog post, we introduce LiteLlama, a lightweight transformer model for efficient language modeling. LiteLlama is designed to be more efficient than the original transformer model while maintaining competitive performance. We achieve this by reducing the model size and the number of parameters. We also introduce a new position embedding method that is more efficient than the original method. We evaluate LiteLlama on the WikiText-103 dataset and show that it outperforms the original transformer model in terms of speed and memory usage while maintaining competitive performance.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LiteLlama(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LiteLlama, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, num_heads=8, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
```


### Related Work

$$
x, \mathbb{X},  \mathbf{X}
$$

$$
\sum_w \frac{1}{\sqrt{d}} \exp\left(\frac{1}{\sqrt{d}}\right)
$$



## LiteLlama Architecture

### Model Overview
asdfad

### Position Embedding

## Training Details

