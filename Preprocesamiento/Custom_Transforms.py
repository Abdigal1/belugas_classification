import numpy as np
import torch
from torchvision import transforms
from skimage.transform import  resize

class MultiInputToTensor(object):
  def __init__(self,to_normalize_keys=[],to_int_keys=[]):
    self.to_normalize_keys=to_normalize_keys
    self.to_int_keys=to_int_keys
    self.TT=transforms.ToTensor()
  def __call__(self,sample):
    for k in self.to_normalize_keys:
      sample[k]=(self.TT(sample[k])/255).float()
    for k in self.to_int_keys:
      sample[k]=self.TT(sample[k]).float()
    return sample

class MultiInputResize(object):
  def __init__(self,shape=(1670,2151),rgb_keys=[],mask_keys=[]):
    self.shape=shape
    self.rgb_keys=rgb_keys
    self.mask_keys=mask_keys
  def __call__(self,sample):
    for k in self.rgb_keys:
      sample[k]=resize(sample[k],self.shape)
    for k in self.mask_keys:
      sample[k]=np.round(resize(sample[k],self.shape,preserve_range=True))
    return sample

class SelectInput(object):
  def __init__(self,output_keys=[]):
    self.output_keys=output_keys
  def __call__(self,sample):
    output={}
    for k in self.output_keys:
      output[k]=sample[k]
    return output