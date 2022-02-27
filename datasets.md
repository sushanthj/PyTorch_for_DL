---
layout: page
title: Datasets
permalink: /Datasets/
nav_order: 6
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Dataset examples

To create our own dataset, we need to subclass the 'Dataset' class from torch.utils.data
**The torch.utils.data also has an in-built DataLoader**

Main Points to note:
- Each dataset class we define customly, should have the functions **__init__**, **__len__**, and **__getitem__**

- Each transformer class we define must have **__init__** and **__call__** functions

## Simple Dataset

Refer this link to understand basic datasets (and torchvision.transforms on this dataset): [simple_dataset](/ref_code/simple_dataset.ipynb)

## Image Dataset

1. [image_dataset_from_scratch](/ref_code/image_dset_from_scratch.ipynb)
2. [image_dataset_pre_built](/ref_code/image_dset_built.ipynb)

## Sample Datasets

Torchvision has some freely avialable datasets which one can use for a cursory evaluation of a model

```python
import torchvision.datasets as dsets

dataset = dsets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
```

*Note: in the above dataset, if we say train=False, then we are using the test dataset (smaller set)*

*Side Note: Usually train, dev and test are split in 80,10,10 ratio from the total no. of images*
