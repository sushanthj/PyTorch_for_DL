---
layout: page
title: PyTorch Transforms and Functionals
permalink: /transforms_functionals/
nav_order: 5
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>


All transforms come under the module **torchvision.transforms** and can be used within the layer of a neural net

we usually import this as:
```python
import torchvision.transforms as T
```

# Basic operations without transforms

## Changing Image Channels (aka flipping elements in any specified dimension)

The way an image is loaded varies with implementations:
- pytorch: CHW (channels, width, height)
- opencv: HWC (height, width, channels)
- numpy: WHC (width, height, channels) *this is also what imagej shows*

FYI, **loading an image in opencv loads it as BGR channels** \
To correct this we usually do cv2.cvt_Color(image, cv2.BGR2RGB)

However, if we need to do the same in numpy or on torch, we essentially just need to flip the dimensions of channels B and R (or we just say flip dimension no. 0 of image tensor)

```python
import torch

flipped_image = torch.flip(input_tensor,[0]) # as CHW, C is in dimension [0]
```

# Tensor operations using torchvision.transform

## Padding

There are two ways of padding:

### Padding using a mask tensor

Lets say we have an image of size (CHW) = (3, 1200, 1300) \
Now let's assume we need to convert it into a square image

*Note. I've used padding on a rectangular image mostly in order to maintain aspect ratio during resize operations*

#### Create a zero mask

```python
import torch

pad_mask = torch.as_tensor(torch.zeros(3,1300,1300, dtype=torch.int8))
```

#### Copy the original image onto the mask

Now the original image was of size:
- orig image: (3,1200,1300)
- pad_mask: (3,1300,1300)

Therefore, we see that we need to only pad 50 pixels above and below in orig image

```python
pad_mask[:,50:1250,:] = orig_image
```

pad_mask will be our final padded image now

### Padding using a torch transform

```python
import torch
import torchvision.transforms as T

# we pass a tuple to the pad function below where (x,y)
# x = pad amount in left and right
# y = pad amount in top and bottom
padder = T.Pad((0,50))

padded_img = padder(input_image)
```

## Resizing

```python
import torchvision.transforms as T

# define the interpolation method
# we're using NEAREST as default BILINEAR is not supported during ONNX conversion
interpolation = T.InterpolationMode.NEAREST

resizer = T.Resize((640,640), interpolation=interpolation)
resized_image = resizer(input_image)
```

# Torch Functionals

Many common tensor operations (used in DNNs) and few other meta operations are bundled in a class called **Module**

Let's create a custom layer by subclassing the Module class (standard procedure)

```python
import torch.nn as nn

class Custom_Layer(nn.Module):
  
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,3)
    
    def forward(self,input):
        x = self.linear(input)
        return x
```

Here we saw that **nn.Linear** was used, but there are also other functions like \
nn.Conv2D and nn.ReLu which are commonly used

These functions like nn.ReLu are said to be of the form **torch.nn.functional**

i.e. Relu and Conv2D are all **torch functionals**

## Using torchvision.transformations inside functionals

Let's club together resizing and padding and put them in a functional called **Sequential**

As the name suggests, Sequential executes anhy operations one after another

```python
import torchvision.transforms as T
import torch

interpolation = T.InterpolationMode.NEAREST
transformer = torch.nn.Sequential(T.Pad((0,pad_const)),T.Resize((640,640),interpolation=interpolation))

output_image = transformer(input_image)
```

## Creating Meta Models

Let's say we have an existing model in pytorch called *pretrained_model* \
and we want to add additional layers to it in the beginning or the end

Usually a model has a few class variables as well like stride, no. of classes etc. We will have to retain this info in our meta model as well

Now let's define our custom model:

```python
import torch
import torchvision.transforms as T

class Custom_Layer(nn.Module):
    # Custom layer for preprocessing
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        input_tensor = input_tensor.type(torch.cuda.FloatTensor)
        #print("input tensor type: ", input_tensor.dtype)
        pad_mask = torch.as_tensor(torch.zeros(3, 1328, 1328, dtype=torch.float32), device=0)
        
        flipped_image = torch.flip(input_tensor,[1])
        pad_mask[:,64:1264,:] = flipped_image
    
        interpolation = T.InterpolationMode.NEAREST
        #transformer = torch.nn.Sequential(T.Pad((0,pad_const)),T.Resize((640,640),interpolation=interpolation))
        transformer = torch.nn.Sequential(T.Resize((640,640),interpolation=interpolation))
        transformed_tensor = transformer(pad_mask).unsqueeze(0)
        return transformed_tensor

class Custom_Model(nn.Module):
    def __init__(self, pretrained_model, nc, names, stride):
        super(Custom_Model, self).__init__()
        self.nc = nc
        self.names = names
        self.preproc_layers = Custom_Layer()
        self.pretrained = pretrained_model
        self.stride = stride
    
    def forward(self, input):
        # add with no grad condition for the next line?
        mod = self.preproc_layers(input)
        mod = self.pretrained(mod)
        return mod
```

We have defined in the above code a custom layer and a custom model.

You may have observed that we used torch.cuda.float16. This is because the existing tensors of the pretrained_model are all loaded onto GPU. To interface with them, we need to initialize our new tensors on the GPU as well.

Now, let's call this custom model in a seperate function:

```python
def custom_load(weights_file_path, device):
    existing_model, stride = attempt_load(weights=weights_file_path, map_location=device, inplace=True, fuse=True)
    nc_existing, names_existing = existing_model.nc, existing_model.names
    extended_model = Custom_Model(pretrained_model=existing_model, nc=nc_existing, names=names_existing, stride=stride)
    return extended_model
```