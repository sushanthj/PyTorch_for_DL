---
layout: page
title: PyTorch Tensors
permalink: /Tensors/
nav_order: 3
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Basics

In PyTorch all parts of a neural net like weights, datasets, biases, outcomes and FC layers are treated as tensors.

One can say that a tensor is a very generalized term for any unit of data in a neural net which is capable of undergoing math operations (like vector ops)

## Creating Tensors

All tensors are created in the form CHW (Channel, Height, Width)

```python
import torch

# three ways of initializing tensors
a = torch.tensor([7,4,3,2,6])
b = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.int32)
c = torch.FloatTensor([0,1,2])
```

```
>> a[0] = 7
>> a.dtype = torch.int64
>> a.type = LongTensor
>> a.size() = torch.tensor(1,5)  # you can also try a.ndimension()
```
## Resizing Tensors
Note in the last line of code above we saw that tensor 'a' had 1 row and 5 columns

Now to resize we do:

```python
a_col = a.view(5,1) # the arguments of view dictate the no. of rows and columns respectively

a_col_auto = a.view(-1,1) # -1 infers no. of rows after readjusting the second dim.
```

# Numpy and Tensor relations
Preferably don't use numpy and work only with tensors in DL networks as it messes up with ONNX or TRT conversions

We will use the function torch.from_numpy and a.numpy()

```python
import numpy as np
import torch

a = np.array([1,2,3,4,5])

torch_tensor = torch.from_numpy(a)
numpy_array = torch_tensor.numpy()
ordinary_list = torch_tensor.to_list()
```

# Broadcasting

Some torch functions can be broadcast along all elements in a tensor or a list

```python
import torch
import numpy as np

a = torch.tensor([0, np.pi/2, 0])

a = torch.sin(a)

>> a = [0,1,0]
```

## Creating an evenly spaced torch tensor

We can use the broadcasting feature of a tensor as we described above along with a torch function called <i> linspace

Note. linspace is also available in numpy with the function having the same name
```python
import torch

a = torch.linspace(-2,2,num=5)

>> a = [-2, -1, 0, 1, 2]
```

# 2D and 3D Tensors and subscripting

In matrices or numpy arrays, accessing individual elements is often called **subscripting**

if 'a' is a 2D tensor, then the individual elements can be accesses as:

```python
a = [...]
>> a[0,2] = # 0th row and 2nd column element
```

# Converting Tensors from one type to another

![](/images/tensor_conversion.jpeg)