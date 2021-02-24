# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:49:55 2021

@author: sefa
"""


import os

import numpy as np
from PIL import Image

import torch
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils



def convert_image_color(image, red=True):
    """Converts image to either red or green"""

    if red:
      image[1:, :, :] *= 0
    else:
      image[::2, :, :] *= 0
    return image


class GenderDataset(datasets.VisionDataset):

  def __init__(self, root='./data', train=True, transform=None, target_transform=None):
    super(GenderDataset, self).__init__(root, transform=transform,
                                target_transform=target_transform)
    
    self.train = train
    self.prepare_converted_dataset()
    if self.train:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'train', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'train', 'train2.pt'))
    else:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'test', 'test.pt')) 


  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_converted_dataset(self):
    if self.train:
        data_dir = os.path.join(self.root, "train")
    else:
        data_dir = os.path.join(self.root, "test")
    if (os.path.exists(os.path.join(data_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(data_dir, 'train2.pt'))) \
        or os.path.exists(os.path.join(data_dir, 'test.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Converted Images for Gender Classification')
    trainset= datasets.ImageFolder(data_dir)

    trainset_1 = []
    trainset_2 = []
    testset = []
    for idx, (im, label) in enumerate(trainset):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(trainset)}')
      im = im.resize((112,112))
      im_array = np.array(im)

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        label = label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = label == 0
      
      if self.train:
          # Flip the color with a probability e that depends on the environment
          if idx%2 == 0:
            # 20% in the first training environment
            if np.random.uniform() < 0.2:
              color_red = not color_red
          else:
            # 10% in the first training environment
            if np.random.uniform() < 0.1:
              color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red

      colored_arr = convert_image_color(im_array, red=color_red)
      
      if self.train:
          if idx%2 == 0:
            trainset_1.append((Image.fromarray(colored_arr), label))
          else:
            trainset_2.append((Image.fromarray(colored_arr), label))
      else:
        testset.append((Image.fromarray(colored_arr), label))

    if self.train:
        torch.save(trainset_1, os.path.join(data_dir, 'train1.pt'))
        torch.save(trainset_2, os.path.join(data_dir, 'train2.pt'))
    else:
        torch.save(testset, os.path.join(data_dir, 'test.pt'))