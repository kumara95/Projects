import pywt
import torch
from torch import nn,optim
import dyadup
from dyadup import dyadup,dyadup1
import numpy as np
from torch import nn,optim
import cupy # import cupy, if available

class Encoder_block(nn.Module):
    def __init__(self):
        super(Encoder_block,self).__init__()
        self.conv1=nn.Sequential(nn.Conv1d(1,2,kernel_size=5,stride=1,padding=2),
                                 nn.Tanh(),
                                 nn.Conv1d(2,2,kernel_size=5,stride=1,padding=2),
                                 nn.Tanh(),
                                 nn.Conv1d(2,2,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.Conv1d(2,2,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.MaxPool1d(4,stride=4),
                                 nn.Conv1d(2,2,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.Conv1d(2,2,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.Conv1d(2,2,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.Conv1d(2,2,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.MaxPool1d(4,stride=4),
                                 nn.Conv1d(2,3,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.Conv1d(3,3,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.Conv1d(3,3,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.Conv1d(3,3,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.Conv1d(3,3,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh(),
                                 nn.MaxPool1d(2,stride=2))
        self.linear=nn.Sequential(nn.Linear(192,70),
                                 nn.Tanh(),
                                 nn.Linear(70,64))
    def forward(self,x):
        out=self.conv1(x)
        h1=torch.reshape(out,[out.shape[0],192])
        out2=self.linear(h1)                     
        return out2                     


class Wavelet_Decoder_block(object):
    def __init__(self):
        super(Wavelet_Decoder_block,self).__init__()
    
    def Train_forward(self,x):
        x=np.reshape(x,[x.shape[0],1,64])
        out1=dyadup1(x,0)
        out2=self.filters(out1)
        
        out3=dyadup1(out2,0)
        out4=self.filters(out3)
        
        out5=dyadup1(out4,0)
        out6=self.filters(out5)
        #print(out6.shape)
        out7=dyadup1(out6,0)
        out8=self.filters(out7)
        
        out8=dyadup1(out8,0)
        out8=self.filters(out8)
      
        out8=np.reshape(out8,[out8.shape[0],out8.shape[2]])
        out8=np.pad(out8,((0,0),(0,31)),'constant')
        out8=np.reshape(out8,[out8.shape[0],1,out8.shape[1]])
        return out8
    def filters(self,x):
        [b,c,v]=x.shape
        out=[]
        wavelet = pywt.ContinuousWavelet('mexh')
        [psi,x1]=wavelet.wavefun(level=3)
        #('mexh',filter_bank='True')
        [dec_lo,dec_hi,rec_lo,rec_hi]=pywt.orthogonal_filter_bank(psi)
        for i in range(b):
            a=x[i,0,:]
            #c=np.zeros(a.shape)
            #coeffs=pywt.array_to_coeffs(a,1,output_format='wavedec')
            out.append(np.convolve(a,rec_lo,mode='same'))
        out1=np.array(out)
        out1=np.reshape(out1,[out1.shape[0],1,out1.shape[1]])
        return out1 
    
class Encoder_Decoder(nn.Module):
    def __init__(self):
        super(Encoder_Decoder,self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(1, 2,kernel_size=5, stride=1, padding=2),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(2,affine=False))
        self.mp1= nn.MaxPool1d(4,stride=4)
        self.encoder2=nn.Sequential(nn.Conv1d(2,2,kernel_size=5,stride=1,padding=2),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(2,affine=False))
        self.mp2= nn.MaxPool1d(8,stride=8)
                                    
        self.encoder3=nn.Sequential(nn.Conv1d(2,4,kernel_size=3,stride=1,padding=1),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(4,affine=False))
        self.mp3= nn.MaxPool1d(4,stride=4)
        self.linears=nn.Sequential(nn.Linear(64,0),
                                   nn.Tanh(),
                                   nn.Linear(10,64))
        
        
        self.deconv1=nn.Sequential(nn.ConvTranspose1d(2,8,kernel_size=4,stride=4),
                                   nn.Conv1d(8,4,kernel_size=5,stride=1,padding=2),
                                   nn.Tanh(),
                                   nn.BatchNorm1d(4,affine=False))    

        self.deconv2=nn.Sequential(nn.ConvTranspose1d(4,4,kernel_size=4,stride=4),
                                   nn.Conv1d(4,4,kernel_size=5,stride=1,padding=2),
                                   nn.Tanh(),
                                   nn.BatchNorm1d(4,affine=False))
        
    
        self.deconv3=nn.Sequential(nn.ConvTranspose1d(4,2,kernel_size=4,stride=4),
                                   nn.Conv1d(2,2,kernel_size=5,stride=1,padding=2),
                                   nn.Tanh(),
                                   nn.BatchNorm1d(2,affine=False),
                                   nn.Conv1d(2,1,kernel_size=5,stride=1,padding=2),
                                    nn.Sigmoid())     
        
    def forward(self,x):
        op_en = self.encoder(x)
        openm=self.mp1(op_en)
        
        op_en2=self.encoder2(openm)
        open2m=self.mp2(op_en2)
        
        op_en3=self.encoder3(open2m)
        open3m=self.mp3(op_en3)
        num_s=open3m.shape
        out=torch.reshape(open3m,(num_s[0],64))
        l=self.linears(out)
        num_shape=l.shape
        dec_inp=torch.reshape(l,(num_shape[0],2,32))
        op_de=self.deconv1(dec_inp)
        #print(op_de.shape)
        #op_deup=self.upsamp(op_de)        
        op_de1=self.deconv2(op_de)
        #print(op_de1.shape)
        #op_de1up=self.upsamp(op_de1)      
        op_de2=self.deconv3(op_de1)
        
        return op_de2
    
    