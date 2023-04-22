import torch
from torch import nn,optim
import numpy as np
from torch import nn,optim
from pybaselines import Baseline, utils, smooth
import math
############### Denoising Autoencoder ###########################
class Encoder_Decoder(nn.Module):
    def __init__(self):
        super(Encoder_Decoder,self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(1, 2,kernel_size=5, stride=1, padding=2),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(2,affine=False))
        self.mp1= nn.MaxPool1d(5,stride=5)
        self.encoder2=nn.Sequential(nn.Conv1d(2,2,kernel_size=5,stride=1,padding=2),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(2,affine=False))
        self.mp2= nn.MaxPool1d(5,stride=5)
                                    
        self.encoder3=nn.Sequential(nn.Conv1d(2,4,kernel_size=3,stride=1,padding=1),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(4,affine=False))
        
        self.mp3= nn.MaxPool1d(2,stride=2)
        self.linears=nn.Sequential(nn.Linear(160,104),
                                   nn.LeakyReLU(0.1),
                                   nn.Linear(104,80))
        
        
        self.deconv1=nn.Sequential(nn.ConvTranspose1d(2,8,kernel_size=2,stride=2),
                                     nn.Conv1d(8,4,kernel_size=5,stride=1,padding=2),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(4,affine=False)) 

        self.deconv2=nn.Sequential(nn.ConvTranspose1d(4,4,kernel_size=5,stride=5),                                     
                                      nn.Conv1d(4,4,kernel_size=5,stride=1,padding=2),
                                      nn.Tanh(),
                                      nn.BatchNorm1d(4,affine=False))
    
        self.deconv3=nn.Sequential(nn.ConvTranspose1d(4,2,kernel_size=5,stride=5),
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
        out=torch.reshape(open3m,(num_s[0],160))
        l=self.linears(out)
        num_shape=l.shape
        dec_inp=torch.reshape(l,(num_shape[0],2,40))
        op_de=self.deconv1(dec_inp)
        #print(op_de.shape)
        #op_deup=self.upsamp(op_de)        
        op_de1=self.deconv2(op_de)
        #print(op_de1.shape)
        #op_de1up=self.upsamp(op_de1)      
        op_de2=self.deconv3(op_de1)
        
        return op_de2
 

####################### Physics informed decoder ####################################

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(1, 2,kernel_size=5, stride=1, padding=2),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(2,affine=False))
        self.mp1= nn.MaxPool1d(5,stride=5)
        self.encoder2=nn.Sequential(nn.Conv1d(2,2,kernel_size=5,stride=1,padding=2),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(2,affine=False))
        self.mp2= nn.MaxPool1d(5,stride=5)
                                    
        self.encoder3=nn.Sequential(nn.Conv1d(2,4,kernel_size=3,stride=1,padding=1),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(4,affine=False))
        
        self.mp3= nn.MaxPool1d(4,stride=4)
        self.linears=nn.Sequential(nn.Linear(80,64),
                                   nn.LeakyReLU(0.1),
                                   nn.Linear(64,36))
        self.batch_size=256
        
    def forward(self,x):
        op_en = self.encoder(x)
        openm=self.mp1(op_en)
        
        op_en2=self.encoder2(openm)
        open2m=self.mp2(op_en2)
        
        op_en3=self.encoder3(open2m)
        open3m=self.mp3(op_en3)
        
        num_s=open3m.shape
        out=torch.reshape(open3m,(num_s[0],80))
        
        l=self.linears(out)
        
        return l
    
  
    #    return out
    def _ngaussian(self, amps, cens,sigmas):
        
        s=torch.zeros(self.batch_size,12,2000).cuda()
        t=torch.linspace(0,2000,2000)
        t1=torch.tile(t,(self.batch_size,1)).cuda()
        #amps=amps.detach().cpu()
        #cens=cens.detach().cpu()
        #sigmas=sigmas.detach().cpu()
        
        for j in range(self.batch_size):
            for i in range (0,12):
                s[j,i,:]=amps[j,i]*(1/(sigmas[j,i]*((2*3.14)**0.5)))*(torch.exp((-1.0/2.0)*((
                    (t1[j,:]-cens[j,i])/sigmas[j,i])**2))).cuda()
        out=torch.sum(s,axis=1)
        return out.cuda()
   
    def Trainforward(self,x,baseline):
        l=self.forward(x)
        out=self._ngaussian(l[:,:12],l[:,12:24],l[:,24:36])
        out=out.reshape([self.batch_size,1,2000])
        spectrum_p=out+baseline.cuda()
        return spectrum_p.float()
    
       

        
    def snip(self,x):
        transformed_data = torch.log(torch.log(torch.sqrt(x + 1) + 1) + 1)
        td=transformed_data.detach().cpu().numpy()
        bkg_4=np.zeros([self.batch_size,1,2000])
        for i in range (self.batch_size):
            f=td[i,0,:]
            f1=f.reshape([2000,1])
            base_l=smooth.snip(f1, max_half_window=40,decreasing=False,smooth_half_window=0,filter_order=8)[0]
            bkg_4[i,0,:]=base_l
        bk=torch.from_numpy(bkg_4)
        baseline = -1 + (torch.exp(torch.exp(bk) - 1) - 1)**2
        return baseline
   
   
    def Trainforwards(self,x):
        l=self.forward(x)
        op=self.snip(x)
        out=self._ngaussian(l[:,:12],l[:,12:24],l[:,24:36])
        out=out.reshape([self.batch_size,1,2000])
        spectrum_p=out+op.cuda()
        return spectrum_p.float()
    
       

        
        
