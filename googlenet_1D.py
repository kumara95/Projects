## 1D Googlenet version first draft
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
__all__ = ['GoogleNet_1D']
class CReLU(nn.Module):
    def __init__(self, inplace=False):
        super(CReLU, self).__init__()
    def forward(self, x):
        x = torch.cat((x,-x),1)
        return F.relu(x) 
class Inception_1d(nn.Module):
  
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()
        self.activation=CReLU()
        #1x1conv branch
        self.b1 = nn.Sequential(nn.Conv1d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm1d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 =nn.Sequential(nn.Conv1d(input_channels, n3x3_reduce, kernel_size=3, padding=1),
                               nn.BatchNorm1d(n3x3_reduce))
        #nn.Sequential(nn.Conv1d(input_channels, n3x3_reduce, kernel_size=1),
                  #              nn.BatchNorm1d(n3x3_reduce),
                   #             nn.ReLU(inplace=True),
                   

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(nn.Conv1d(input_channels, n5x5_reduce, kernel_size=5,padding=2),
                                nn.BatchNorm1d(n5x5_reduce))
            #nn.ReLU(inplace=True),
            #nn.Conv1d(n5x5_reduce, n5x5_reduce, kernel_size=5, padding=2),
            #nn.BatchNorm1d(n5x5_reduce))

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(nn.MaxPool1d(3, stride=1,padding=1),
                                nn.Conv1d(input_channels, pool_proj, kernel_size=1),
                                nn.BatchNorm1d(pool_proj),
                                nn.ReLU(inplace=True))
     

    def forward(self,x):
        x1=self.b1(x)
        #print(x1.shape)
        x2=self.b2(x)
        x2=self.activation(x2)
        #print(x2.shape)
        x3=self.b3(x)
        x3=self.activation(x3)
        #print(x3.shape)
        x4=self.b4(x)
        #print(x4.shape)
        return torch.cat([x1, x2, x3, x4], dim=1)

#introduce a class for googlenet
#nn module is a pytorch class; googlenet has been drived from nn.module
class googlenet_1D(nn.Module):

    def __init__(self, num_classes=16, backbone_fc=False):
                                
        super().__init__()
        self.activation=CReLU()
        self.prelayer = nn.Sequential(nn.Conv1d(1, 8, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(8))
        self.prelayer2=nn.Sequential(nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm1d(16))
        self.prelayer3=nn.Sequential(nn.Conv1d(32, 8, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(8),
                                     nn.ReLU(inplace=True))
        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception_1d(input_channels=8, n1x1=16, n3x3_reduce=8, n3x3=8, n5x5_reduce=8, n5x5=8,
                               pool_proj=16)
        self.maxpool = nn.MaxPool1d(3, stride=3)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.conv1 =nn.Sequential(nn.Conv1d(64,16,kernel_size=1,stride=1),
                                  nn.ReLU(inplace=True))
        self.conv2=nn.Sequential(nn.Conv1d(16,32,kernel_size=3,stride=1,padding=1),
                                  nn.ReLU(inplace=True))
        self.conv3=nn.Sequential(nn.Conv1d(16,32,kernel_size=3,stride=1,padding=1),
                                  nn.ReLU(inplace=True))
        self.maxpool = nn.Sequential(nn.MaxPool1d(3, stride=3),
                                     nn.Conv1d(64,21,kernel_size=1),
                                     nn.ReLU(),
                                     nn.Conv1d(21,64,kernel_size=3,padding=1),
                                     nn.ReLU())
                                
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(64, num_classes)
        self.log_softmax = nn.LogSoftmax(dim = 1)

    def forward(self,x):
        #print(x.size())
        x = self.prelayer(x)
        x=self.activation(x)
        x = self.prelayer2(x)
        x=self.activation(x)
        x = self.prelayer3(x)
        #x=self.activation(x)
      
        x = self.a3(x)
                                
        x=self.maxpool(x)
        ####### BSM module (kind of inception)
        x = self.conv1(x)
        x1  = self.conv2(x)
        x2=self.conv3(x)
        x=torch.cat([x1,x2],dim=1)
        ####### ACM module
        x=self.maxpool(x)
        
    

        x = self.avgpool(x)
        #print(x.size())
        #x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x=self.log_softmax(x)

        return x

def main():
    googlenet1D=googlenet_1D(16)
    googlenet1D.eval()
    with torch.no_grad():
        inputs=torch.tensor(np.ones((10,1,2048),dtype=np.float32))
        activation_vectors = googlenet1D(inputs)


if __name__ == '__main__':
    main()


