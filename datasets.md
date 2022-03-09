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