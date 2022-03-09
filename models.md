---
layout: page
title: PyTorch Models
permalink: /torch.nn/
nav_order: 8
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Creating Simple Models using torch.nn

Even the simplest linear regression problem has to be bundled in a model in PyTorch.

PyTorch allows for creating simple linear regression models (which are of the form y = b + wx) in a straightforward manner called *nn.Linear*

We will make a custom model which does only linear regression in the torch way:

```python
import torch
from torch import nn

# Customize Linear Regression Class

class LR(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out
```

Now we can create an object of the above calss, try to feed it single or multiple inputs
```python

# Create the linear regression model. Print out the parameters.
lr = LR(1, 1)
print("The parameters: ", list(lr.parameters()))
print("Linear model: ", lr.linear)

# Try our customize linear regression model with single input
x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)

# Try our customize linear regression model with multiple input
x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)
```

**Note in any model which is constructed using the nn.Module class, you do not need to specify the forward function, you just need to call the custom model class as: *output = Custom_Model(input)***

## Accessing model params

Using the above model class as an example, we can get the model weights and biases by:
```python
model = custom_model(input)
print(list(model.parameters()))
```

We can also access these weights and biases and even modify them using **model.state_dict()**

```python
model = custom_model(input)
# init weights for the layer 'linear'
model.state_dict()['linear.weight'].data[0] = torch.tensor([0.51])
model.state_dict()['linear.bias'].data[0] = torch.tensor([0.4])

print(model.state_dict())
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

# Model Training

