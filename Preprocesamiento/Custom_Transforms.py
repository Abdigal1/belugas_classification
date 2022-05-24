import numpy as np
import torch
from torchvision import transforms
from skimage.transform import  resize

class MultiInputToTensor(object):
  def __init__(self,images=["x"],metadata=["y","vp"]):
    self.images=images
    self.metadata=metadata
    self.TT=transforms.ToTensor()
  def __call__(self,sample):
    for k in self.images:
      sample[k]=(self.TT(sample[k])/255).float()
    for k in self.metadata:
      sample[k]=torch.tensor(sample[k]).float()
    return sample

class vp_one_hot_encoding(object):
  def __init__(self):
    self.encoding={
      'left':np.array([1,0,0]),
      'right':np.array([0,1,0]),
      'top':np.array([0,0,1])
    }
    pass
  def __call__(self,sample):
    sample["vp"]=self.encoding[sample["vp"]]
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