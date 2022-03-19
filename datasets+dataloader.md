---
layout: page
title: Datasets & Dataloaders
permalink: /Datasets+Dataloaders/
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

## Adding preprocessing to a model: the dataloader way

The basic layout of a dataset class is shown below. As mentioned earlier it needs to have three basic functions of init, len, getitem:

```python
# Define class for dataset

class toy_set(Dataset):
    
    # Constructor with defult values 
    def __init__(self, length = 100, transform = None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
     
    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    # Get Length
    def __len__(self):
        return self.len
```

Now, we can add some preprocessing to the dataset loader by giving a transformer as an argument to the dataset class:

```python
# Combine two transforms: crop and convert to tensor. Apply the compose to toy_set

croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = toy_set(transform=croptensor_data_transform )
print("The shape of the first element tensor: ", dataset[0][0].shape)
```

## Load images from a folder and image_names in csv as a dataset:

```python
class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        
        # Image directory
        self.data_dir=data_dir
        
        # The transform is goint to be used on image
        self.transform = transform
        data_dircsv_file=os.path.join(self.data_dir,csv_file)
        # Load the CSV file contians image info
        self.data_name= pd.read_csv(data_dircsv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0] 
    
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        
        # Image file path
        img_name=os.path.join(self.data_dir,self.data_name.iloc[idx, 1])
        # Open image file
        image = Image.open(img_name)
        
        # The class label for the image
        y = self.data_name.iloc[idx, 0]
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
```

Creating a object of dataset class:
```python
dataset = Dataset(csv_file=csv_file, data_dir=directory)
```

# Dataloader

Both Dataset and Dataloader are inbuilt classes in pytorch

We use both in conjunction in cases such as in Stochastic Gradient Descent where each \
datapoint from the dataset is executed one by one (instead of a bunch of datapoints \
such as in batch gradiet descent)

See the example of SGD below where we define a **dataset and dataloader**:

```python
from torch.utils.data import Dataset, DataLoader

# Dataset Class
class Data(Dataset):
    
    # Constructor
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 1 * self.x - 1
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self,index):    
        return self.x[index], self.y[index]
    
    # Return the length
    def __len__(self):
        return self.len


# Create the dataset and check the length
dataset = Data()
print("The length of dataset: ", len(dataset))

# Create DataLoader
trainloader = DataLoader(dataset = dataset, batch_size = 1)


# Training the model
w = torch.tensor(-15.0,requires_grad=True)
b = torch.tensor(-10.0,requires_grad=True)
LOSS_Loader = []

def train_model_DataLoader(epochs):
    
    # Loop
    for epoch in range(epochs):
        
        # SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
        Yhat = forward(X)
        
        # store the loss 
        LOSS_Loader.append(criterion(Yhat, Y).tolist())
        
        for x, y in trainloader:
            
            # make a prediction
            yhat = forward(x)
            
            # calculate the loss
            loss = criterion(yhat, y)
            
            # Backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            
            # Updata parameters slope
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr* b.grad.data
            
            # Clear gradients 
            w.grad.data.zero_()
            b.grad.data.zero_()

train_model_DataLoader(10)
```

# Mini Batches

Consider the below example:
![](/images/batch_size.jpeg)

The logic is that one epoch means one complete sweep of the datapoints present in the dataset

Therefore, if a dataset has only 100 images and our mini batch size is 50, then we will need 2 iterations of the train loop to finish 1 epoch.

No. of iterations per epoch = (Dataset size / batch size)

## Side Note:

A datapoint is an interesting term in Data science. It may not mean one complete object in a dataset.

Say we have an image with two plants in them. Then one datapoint refers not to one image, but to on box in the image (which contains the plant).