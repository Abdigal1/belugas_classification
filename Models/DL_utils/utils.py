from torch import dropout, nn
import torch.nn.functional as F
import torch


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w

def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1,out_pad=0):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
        
    if type(dilation) is not tuple:
        dilation = (dilation, dilation)
        
    if type(out_pad) is not tuple:
        out_pad = (out_pad, out_pad)
        
    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + dilation[0]*(kernel_size[0]-1) + out_pad[0]+1
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + dilation[0]*(kernel_size[1]-1) + out_pad[1]+1
    
    return h, w

class s_view(nn.Module):
    def forward(self,x):
        if len(x.shape)==4:
            self.i_shape=x.shape
            out=x.view(x.shape[0],-1)
        elif len(x.shape)==2:
            out=x.view(self.i_shape)
        return out

class NN_layer(nn.Module):
    def __init__(self,inp,out,act=nn.ReLU(),batch_norm=True,dropout=None):
        super(NN_layer,self).__init__()
        self.batch_norm=batch_norm
        self.layer=nn.ModuleList(
            [nn.Linear(inp,out)]+([nn.BatchNorm1d(out)] if self.batch_norm else [])+[act]+([nn.Dropout(dropout)] if dropout!=None else [])
            )
    def forward(self,x):
        for sl in self.layer:
            x=sl(x)
        return x


class NeuralNet(nn.Module):
    def __init__(self,layer_sizes=[300,150,50],activators=nn.LeakyReLU(),batch_norm=True,dropout=None):
        super(NeuralNet,self).__init__()
        self.layer_sizes=layer_sizes
        #self.activators=activators

        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(layer_sizes)-1)]

        if isinstance(dropout,float):
            self.dropout=[dropout for i in range(len(layer_sizes)-1)]
        else:
            self.dropout=[None for i in range(len(layer_sizes)-1)]

        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(layer_sizes)-1)]
        else:
            self.activators=activators

        self.layers=nn.ModuleList(
            [
                nn.Sequential(NN_layer(in_size,out_size,act,bat_norm,dropout))
                for in_size,out_size,act,bat_norm,dropout in zip(
                    self.layer_sizes[:-1],
                    self.layer_sizes[1:],
                    self.activators,
                    self.batch_norm,
                    self.dropout
                )
            ]
        )
    def forward(self,x):
        for l in self.layers:
            x=l(x)
        return x

class set_conv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,act=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
        super(set_conv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)

        self.comp_layer=nn.ModuleList(
            [nn.Conv2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding)]+\
            ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])+\
            [act]+\
            ([nn.Dropout(dropout)] if dropout!=None else [])+\
            ([nn.MaxPool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)] if pooling else [])
        )

    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x

class set_deconv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,act=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
        super(set_deconv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
            self.out_pad=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)
            self.out_pad=1

        self.comp_layer=nn.ModuleList(
            [nn.ConvTranspose2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding,output_padding=self.out_pad)]+\
            ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])+\
            [act]+\
            ([nn.Dropout(dropout)] if dropout!=None else [])+\
            ([nn.MaxUnpool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)] if pooling else [])
        )
    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x


class b_encoder_conv(nn.Module):
    def __init__(self,repr_sizes=[3,32,64,128,256],
                kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
        super(b_encoder_conv, self).__init__()
        self.repr_sizes=repr_sizes
        self.stride=[stride for i in range(len(repr_sizes)-1)]
        
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes)-1)]
        else:
            self.kernels=kernel_size
        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(repr_sizes)-1)]
        else:
            self.activators=activators
        #pooling
        if isinstance(pooling,bool):
            self.pooling=[pooling for i in range(len(repr_sizes)-1)]
        else:
            self.pooling=pooling
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes)-1)]
        else:
            self.batch_norm=batch_norm
        
        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(repr_sizes)-1)]
        else:
            self.dropout=dropout
        
        self.im_layers=nn.ModuleList(
            [
                set_conv(repr_in,
                repr_out,
                kernel_size,
                act,
                pooling,
                batch_norm,
                dropout,
                stride
                )
                for repr_in,repr_out,kernel_size,act,pooling,batch_norm,dropout,stride in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
                    self.kernels,
                    self.activators,
                    self.pooling,
                    self.dropout,
                    self.batch_norm,
                    self.stride
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        return x
    
class b_decoder_conv(nn.Module):
    def __init__(self,repr_sizes=[32,64,128,256],
                kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
        super(b_decoder_conv,self).__init__()
        self.repr_sizes=repr_sizes
        self.repr_sizes=self.repr_sizes[::-1]
        self.stride=[stride for i in range(len(repr_sizes)-1)]
        
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes)-1)]
        else:
            self.kernels=kernel_size
        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(repr_sizes)-1)]
        else:
            self.activators=activators
        #pooling
        if isinstance(pooling,bool):
            self.pooling=[pooling for i in range(len(repr_sizes)-1)]
        else:
            self.pooling=pooling
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes)-1)]
        else:
            self.batch_norm=batch_norm

        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(repr_sizes)-1)]
        else:
            self.dropout=dropout
        
        self.im_layers=nn.ModuleList(
            [
                set_deconv(repr_in,
                repr_out,
                kernel_size,
                act,
                pooling,
                batch_norm,
                dropout,
                stride
                )
                for repr_in,repr_out,kernel_size,act,pooling,batch_norm,dropout,stride in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
                    self.kernels,
                    self.activators,
                    self.pooling,
                    self.batch_norm,
                    self.dropout,
                    self.stride
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        return x