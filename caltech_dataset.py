from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import numpy as np


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        r=root.split("/")                   
        input=[]
        with open(r[0]+"/"+split+".txt", 'r') as f:
            input = f.readlines()

        to_keep=[]
        for line in input:
          if line.find("BACKGROUND") < 0: #non è presente
            to_keep.append(line.rstrip("\n"))

        labels_cat=[]
        data=[]
        for folder in os.listdir(root):
          #print(folder)
          if folder.find("BACKGROUND")<0:
            for img in os.listdir(root+"/"+folder):
                #print(img)
                p=folder+"/"+img
                if p in to_keep:
                    labels_cat.append(folder)
                    image=root+"/"+folder+"/"+img
                    normale=pil_loader(image)
                    data.append(normale) #1
                    rotated=rotate(image, angle=45, mode = 'wrap')
                    data.append(rotated) #2
                    labels_cat.append(folder)
                    transform = AffineTransform(translation=(25,25))
                    wrapShift = warp(image,transform,mode='wrap')
                    data.append(wrapShift) #3
                    labels_cat.append(folder)
                    flipLR = np.fliplr(image) #4
                    data.append(flipLR)
                    labels_cat.append(folder)
                    flipUD = np.flipud(image) #5
                    data.append(flipUD)
                    labels_cat.append(folder) #6
                    sigma=0.155
                    #add random noise to the image
                    noisyRandom = random_noise(image,var=sigma**2)
                    data.append(noiseRandom) #7
                    labels_cat.append(folder) 
                    blurred = gaussian(image,sigma=1,multichannel=True)
                    data.append(blurred) #8
                    labels_cat.append(folder)

        labels=[]
        el_root=[i for i in os.listdir(root) if i.find("BACKGROUND")<0]
        tot_cat=sorted(el_root)
        for lc in labels_cat:
          for i,c in enumerate(tot_cat):
            if lc == c:
              labels.append(i)

          
        self.data=data
        self.labels=labels
        self.transform=transform

       # '''
       #- Here you should implement the logic for reading the splits files and accessing elements
       # - If the RAM size allows it, it is faster to store all data in memory
        #- PyTorch Dataset classes use indexes to read elements
        #- You should provide a way for the __getitem__ method to access the image-label pair                        -DONE
         # through the index
        #- Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class)  -DONE
        #'''
        
    def __getitem__(self, index):
        #'''
        #__getitem__ should access an element through its index
        #Args:
         #   index (int): Index

        #Returns:
        #    tuple: (sample, target) where target is class_index of the target class.
        #'''

        #image, label = ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int
        label=self.labels[index]
        image=self.data[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        #'''
        #The __len__ method returns the length of the dataset
        #It is mandatory, as this is used by several other components
        #'''
        length = len(self.labels) # Provide a way to get the length (number of elements) of the dataset
        return length
