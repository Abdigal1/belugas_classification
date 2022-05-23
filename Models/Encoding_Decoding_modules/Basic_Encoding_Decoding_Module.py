from DL_utils.utils import *

class EncodingDecodingModule(nn.Module):
  def __init__(self):
    super(EncodingDecodingModule,self).__init__()
    #Encoding modules
    self.el1=set_conv(1, 2, kernel_size=5, act=nn.ReLU(), pooling=False, batch_norm=False, dropout=None, stride=1)
    self.el2=set_conv(2, 4, kernel_size=5, act=nn.ReLU(), pooling=False, batch_norm=False, dropout=None, stride=1)
    #Decoding modules
    self.dl1=set_deconv(4, 2, kernel_size=5, act=nn.ReLU(), pooling=False, batch_norm=False, dropout=None, stride=1)
    self.dl2=set_deconv(2, 1, kernel_size=5, act=nn.ReLU(), pooling=False, batch_norm=False, dropout=None, stride=1)
    #flatten
    self.fl=s_view()

  def Encoding(self,x):
    ex=self.el1(x)
    ex=self.el2(ex)
    ex=self.fl(ex)
    return ex
    
  def Decoding(self,z):
    dz=self.fl(z)
    dz=self.dl1(dz)
    dz=self.dl2(dz)
    return dz

class Basic_Convolutional_EDM(nn.Module):
  def __init__(self,repr_sizes=[3,32,64,128,256],kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
    super(Basic_Convolutional_EDM,self).__init__()
    #Encoding modules
    self.ENC=b_encoder_conv(
                   repr_sizes=repr_sizes,
                   kernel_size=kernel_size,
                   activators=activators,
                   batch_norm=batch_norm,
                   dropout=dropout,
                   stride=stride,
                   pooling=pooling
                   )
    #Decoding modules
    self.DEC=b_decoder_conv(
                   repr_sizes=repr_sizes,
                   kernel_size=kernel_size,
                   activators=activators,
                   batch_norm=batch_norm,
                   dropout=dropout,
                   stride=stride,
                   pooling=pooling
                   )
    #flatten
    self.fl=s_view()

  def sanity_check(self,x):
    ex=self.ENC(x)
    ex=self.fl(ex).shape
    return ex

  def Encoding(self,x):
    ex=self.ENC(x)
    ex=self.fl(ex)
    return ex
    
  def Decoding(self,z):
    dz=self.fl(z)
    dz=self.DEC(dz)
    return dz