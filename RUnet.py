###############The architecture for RoI aware conditional- variational autoencoder##########
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import torch.distributions as dist

class Encoder_block(nn.Module):
    def __init__(self,latent_dim):
        """
    Initializes the encoder network
    """
        super(Encoder_block, self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(1, 2,kernel_size=5, stride=1, padding=2),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(2,affine=False))
                                   
        self.mp1=nn.MaxPool1d(4,stride=4)
        self.encoder2= nn.Sequential(nn.Conv1d(2, 4,kernel_size=5, stride=1, padding=2),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(4,affine=False))
        self.mp2= nn.MaxPool1d(4,stride=4)
        
        self.encoder3=nn.Sequential(nn.Conv1d(4,4,kernel_size=5,stride=1,padding=2),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(4,affine=False))
        self.mp3=nn.MaxPool1d(4,stride=4)
        
        #Out channel is 3 because we have 3 classes
        self.conv=nn.Conv1d(4,3,kernel_size=5,stride=1,padding=2)
        self.mp=nn.MaxPool1d(64,stride=64)
       

    def forward(self, X,y):
        op_en = self.encoder(X)
        openm=self.mp1(op_en)
        op_en2=self.encoder2(openm)
        open2m=self.mp2(op_en2)
        op_en3=self.encoder3(open2m)
        open3m=self.mp3(op_en3)
        LocOut=self.conv(open3m)
        LocOut=F.softmax(LocOut,dim=1)
        Y=self.mp(y)        
        #####Conditionality part#####        
        return [LocOut,Y,op_en,op_en2,op_en3] 
    
    def trs(self,x):
        op_en = self.encoder(x)
        openm=self.mp1(op_en)
        op_en2=self.encoder2(openm)
        open2m=self.mp2(op_en2)
        op_en3=self.encoder3(open2m)
        open3m=self.mp3(op_en3)
        LocOut=self.conv(open3m)
        #LocOut=F.softmax(LocOut,dim=1)
        return LocOut

        
class ROIposteriorensemble(nn.Module):
    """
    Network to compute posterior for segmented data
    """
    def __init__(self,latent_dim):
        super(ROIposteriorensemble,self).__init__()
        latent=3*32
        self.encoder_mu_seg1=nn.Sequential(nn.Linear(latent,100),
                                          nn.LeakyReLU(0.1),
                                          nn.Linear(100,2*latent_dim))
        #self.encoder_mu_seg2=nn.Sequential(nn.Linear(latent,70),
            #                              nn.LeakyReLU(0.1),
           #                               nn.Linear(70, 2*latent_dim))
        #self.encoder_mu_seg3=nn.Sequential(nn.Linear(latent,70),
         #                                 nn.LeakyReLU(0.1),
          #                                nn.Linear(70, 2*latent_dim))
        
        self.latent_dim=latent_dim
    def forward(self,X,seg=None):   
        #h1=X[:,0,:]
        #h2=X[:,1,:]
        #h3=X[:,2,:]
        latent=3*32
        h1=torch.reshape(X,[X.shape[0],latent])
        #h2=torch.reshape(h2,[h2.shape[0],latent])
        #h3=torch.reshape(h3,[h3.shape[0],latent])
        
        mu=self.encoder_mu_seg1(h1)
        #mu2=self.encoder_mu_seg2(h2)
        #mu3=self.encoder_mu_seg3(h3)
        
        #print(mu.shape)
        #mu_log=torch.squeeze(mu,dim=2)#spatial axes to be defined
        #print(mu_log.shape)
        
        mean=mu[:,:self.latent_dim]
        sigma=mu[:,self.latent_dim:]
        
        #mean1=mu2[:,:self.latent_dim]
        #sigma1=mu2[:,self.latent_dim:]
        
        #mean2=mu3[:,:self.latent_dim]
        #sigma2=mu3[:,self.latent_dim:]
        #sum_mean=mean+mean1+mean2
        #sum_sigma=torch.exp(sigma)+torch.exp(sigma1)+torch.exp(sigma2)
        #    mean, torch.exp(sigma),mean1,torch.exp(sigma1),mean2,torch.exp(sigma2),
        return mean,torch.exp(sigma)
class Classifier(nn.Module):
    def __init__(self, dim):
        super(Classifier, self).__init__()
        self.layer1=nn.Sequential(nn.Linear(16,128),
                                  nn.Tanh(),
                                  nn.Linear(128,256),
                                  nn.Tanh(),
                                  nn.Linear(256,16))
        self.logsoftmax=nn.Softmax(dim=1)

    def forward(self, x):
        out=self.layer1(x)
        y=self.logsoftmax(out)
        return y       
        
        
class CondPrior(nn.Module):
      """
    Network that computes prior for labels
    """
    def __init__(self, dim):
        super(CondPrior, self).__init__()
        self.dim = dim
        self.layer2=nn.Sequential(nn.Linear(16,128),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(128,32))

    def forward(self, x):
       
        out = self.layer2(x)
        loc=out[:,0:16]
        scale = out[:,16:32]
        return loc, torch.exp(scale)
                    
class Localregion_decoder(nn.Module):
    """
    Intializes Segmentation decoder network
    """
    def __init__(self,latentdim):
        super(Localregion_decoder,self).__init__()
        
        self.up_samp1=nn.ConvTranspose1d(4,4,kernel_size=4,stride=4)
        self.deconv1=nn.Sequential(nn.Conv1d(4,4,kernel_size=5,stride=1,padding=2),
                                   nn.Tanh(),
                                   nn.BatchNorm1d(4,affine=False))
        self.deconvf=nn.Sequential(nn.Conv1d(8,4,kernel_size=5,stride=1,padding=2),
                                   nn.Tanh(),
                                   nn.BatchNorm1d(4,affine=False))
        self.up_samp2=nn.ConvTranspose1d(4,2,kernel_size=4,stride=4)
        self.deconv2=nn.Sequential(nn.Conv1d(2,2,kernel_size=5,stride=1,padding=2),
                                   nn.Tanh(),
                                   nn.BatchNorm1d(2,affine=False))
        self.deconvf2=nn.Sequential(nn.Conv1d(4,2,kernel_size=5,stride=1,padding=2),
                                   nn.Tanh(),
                                   nn.BatchNorm1d(2,affine=False))
        self.up_samp3=nn.ConvTranspose1d(2,2,kernel_size=4,stride=4)           
          #For segmentation                      
        self.segtop=nn.Sequential(nn.Conv1d(2,3,kernel_size=5,stride=1,padding=2),
                                  nn.LogSoftmax(dim=1))                               
       
                                
    def forward(self,x1,x2,x3,RoI):
        """
        Computes the RoI tensor pyramid for skip connections from encoder layers
        """
        P_region=[]
        for i in range(len(RoI)):
            X_start=RoI[i][1]
            X_end=RoI[i][2]
            X_start=int(X_start)
            X_end=int(X_end)
            RoITensorPyramid=[x3[:,:,X_start:X_end],
                             x2[:,:,X_start*4:X_end*4],
                             x1[:,:,X_start*16:X_end*16]]
            RoI_1=torch.tensor(RoITensorPyramid[0],dtype=torch.float32)
            RoI_2=torch.tensor(RoITensorPyramid[1],dtype=torch.float32)
            RoI_3=torch.tensor(RoITensorPyramid[2],dtype=torch.float32)
            #Conv1 with upsamp
            up1=self.up_samp1(RoI_1)
            
            #print(x_cat.shape)
            p_cat=self.deconv1(up1)
            x_cat=torch.cat((p_cat,RoI_2),1)
            x_f=self.deconvf(x_cat)
            #Conv2 with upsamp
            p_up=self.up_samp2(x_f)
           
            p1=self.deconv2(p_up)
            p_cat1=torch.cat((p1,RoI_3),1)
            
            x_f1=self.deconvf2(p_cat1)
            #Conv3 with upsamp
            p_up1=self.up_samp3(x_f1)
                   
            #Segmentation
            p_reg=self.segtop(p_up1) #Unwanted we know exactly where to segment
            P_region.append(p_reg)
            
   
        return P_region
                                
    def TrainForward(self,x1,x2,x3,RoIs,y_region):
        """
        Computes the forward operation on the network
        """
        Y_Region=[]
        #Extract in-region labels
        for i in range(len(RoIs)):
            Xstart=RoIs[i][1]
            Xend=RoIs[i][2]
            Xstart=int(Xstart)
            Xend=int(Xend)
            y_region_RoI=y_region[:,:,Xstart*64:Xend*64]
            Y_Region.append(y_region_RoI)
        #Y_r=torch.stack(Y_Region)        
        P_Region=self.forward(x1,x2,x3,RoIs)
        return P_Region,Y_Region
def region_props(spectrum,i):
    """""
    computes region bounding box 
    """"
    
    Bbox=np.zeros([1,3])
    if i==0:
        #Bbox[0]=[0,0,5]  
        Bbox[0]=[0,0,32]  
        
    elif i==1:
        Bbox[0]=[1,0,32]
    else:
        Bbox[0]=[2,0,32]     
    Prop=Bbox
    #Prop=Prop.to('cuda:0')
    return Prop

class Decoder_Block(nn.Module):
    """
    Initializes Reconstruction decoder network
    """
    def __init__(self,latent_dim):
        super(Decoder_Block,self).__init__()
        self.lin=nn.Linear(latent_dim,64)
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
                                    
    
    def forward(self,X):        
        #upx=self.upsamp(X)
        l=self.lin(X)
        num_shape=l.shape
        dec_inp=torch.reshape(l,(num_shape[0],2,32))
        op_de=self.deconv1(dec_inp)        
        op_de1=self.deconv2(op_de)           
        op_de2=self.deconv3(op_de1)
        return op_de2
def compute_kl(locs_q, scale_q, locs_p=None, scale_p=None):
    """
    Computes the KL(q||p)
    """
    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:
        scale_p = torch.ones_like(scale_q)

    dist_q = dist.Normal(locs_q, scale_q)
    dist_p = dist.Normal(locs_p, scale_p)
    return dist.kl.kl_divergence(dist_q, dist_p).sum(dim=-1)

def img_log_likelihood(recon, xs):
        return dist.Laplace(recon, torch.ones_like(recon)).log_prob(xs).sum(dim=(1,2))
    
class RUVAEnet(nn.Module):
    def __init__(self,upscale,z_dim, num_classes,y_prior):
        super(RUVAEnet,self).__init__()
        self.encoder_net=Encoder_block(upscale)
        #self.encoder_vae=EncoderBlock()
        self.n_classes=3
        #May be Region of Interest Pooling, self.roi_pooling=RoIPoolFunction(pool_length, spatial_scale)
        ######################## Decoder_side #########################################
        self.decoder_net=Localregion_decoder(upscale).to('cuda:0')
        
        ###################### Conditionality part#####################
        #Latent_dim from data
        self.z_dim = z_dim
        
        #Latent Dim from number of labels 
        self.z_classify = num_classes
        
        #Number of denotations for each labels
        self.z_style = z_dim - num_classes
        

        self.num_classes = num_classes
        
        #Prior initalization for denotations
        self.ones = torch.ones(1, self.z_style)
        self.zeros = torch.zeros(1, self.z_style)
        self.y_prior_params = y_prior
        
        self.decoder = Decoder_Block(self.z_dim)
        self.classifier = Classifier(self.num_classes)
        
        #
        self.cond_prior = CondPrior(self.num_classes)

        self.ones = self.ones.cuda()
        self.zeros = self.zeros.cuda()
        self.y_prior_params = self.y_prior_params.cuda()
        self.cuda()
        
        ########## Conditionality for segments###############
        #self.pseg_net=ROIposterior(self.z_dim)
        self.pseg_net2=ROIposteriorensemble(self.z_dim)
   
    def Localization(self,LocOut,Train=True): # May be modifying required 
        ##############According to paper this is for getting bounding box pyramid for each of the classes #####################
        ################ The commented parts necessary for the further developments #####################
        
        #This function is used to get the region that are important after extracting features from convolution 
        #so that the network is trained to detect automatically the area #     
        LocOut = LocOut.to(device='cpu')
        LocOut = LocOut.detach().numpy()
        RoIs=[]
        Bbox=[]
        #num=0
    
        for i in range(0,self.n_classes): #self.n_classes=3 #Three segments of the inelastic spectrum
            Heatmap = LocOut[0,i]
            
            #Minmax normalization
            Heatmap = (Heatmap-np.min(Heatmap))/(np.max(Heatmap)-np.min(Heatmap))
            Heatmap[Heatmap<0.5]=0
            Heatmap[Heatmap>=0.5]=1
            
            #Heatmap*=255 Conversion for image
            
            #Measures region properties to get the area of the three segments
            #ConnectMap=label(Heatmap, connectivity= 1)
            #Props = regionprops(ConnectMap)
            #Area=3 # We have 3 regions/classes
            Props=region_props(Heatmap,i)
         
            
            #obtaining region of interests
            ##### Assuming the length of the interested areas is one.
            
            #for j in range(3):
            #Area.append(Props[j]['area'])
            Bbox.append(Props[0,:])
            #Pyramid coordinates starting boundixg box size
            OverDesignRange=[2,2]
            #if Bbox[j][0]-OverDesignRange[0]<0:
             #   Bbox[j][0]=0
            #else:
             #   Bbox[j][0]-=OverDesignRange[0]
                    
           # if Bbox[j][1]+OverDesignRange[1]>=Heatmap.shape[1]-1:
            #        Bbox[j][1]=Heatmap.shape[1]-1
            #else:
             #       Bbox[j][1]+=OverDesignRange[1]           
            
            #Area=np.array(Area)
            #Bbox=np.array(Bbox)
            #argsort=np.argsort(Area)
            #Area=Area[argsort]
            #Bbox=Bbox[argsort]
            #Area=Area[::-1]
            #Bbox=Bbox[::-1,:]
            
            #max_boxes=3
            #if Area.shape[0]>=max_boxes:
            #    OutBbox=Bbox[:max_boxes,:]
            #elif Area.shape[0]==0:
             #   OutBbox=np.zeros([1,2],dtype=np.int)
              #  OutBbox[0]=[0,1]
            #else:
             #   OutBbox=Bbox
        Bbox=np.array(Bbox)
        for j in range(Bbox.shape[0]):
            RoIs.append(Bbox[j,:])
        RoIs=np.array(RoIs)
        return RoIs
        
    def Trainforward(self,x,y_region,label):
        LocOut,Y,x1,x2,x3=self.encoder_net(x,y_region)             
        #####Getting bounding boxes for RoIs naively###########
        RoI=self.Localization(LocOut)
        ###################Conditionality#########
        bs = x.shape[0]
        LocOut1=self.encoder_net.trs(x)  
        post_params=self.pseg_net2(LocOut1,seg=True) #Data is converted to posterior
        z = dist.Normal(*post_params).rsample()
        zc, zs = z.split([self.z_classify, self.z_style], 1)
        #print(zc.shape)
        output=self.classifier(zc)
        qyzc = dist.Bernoulli(logits=output)
        log_qyzc = qyzc.log_prob(label).sum(dim=-1)
        # compute kl
        locs_p_zc, scales_p_zc = self.cond_prior(label)
        prior_params = (torch.cat([locs_p_zc, self.zeros.expand(bs, -1)], dim=1), 
                        torch.cat([scales_p_zc, self.ones.expand(bs, -1)], dim=1))
        #prior_params = (self.zeros.expand(bs, -1), self.ones.expand(bs, -1))
        kl = compute_kl(*post_params, *prior_params)

        #compute log probs for x and y
        recon = self.decoder(z)
        log_qyx = self.classifier_loss(x, label)
        log_pxz = img_log_likelihood(recon, x)

        # we only want gradients wrt to params of qyz, so stop them propogating to qzx
        log_qyzc_ = dist.Bernoulli(logits=self.classifier(zc.detach())).log_prob(label).sum(dim=-1)
        z1=self.y_prior_params.expand(bs, -1)
        log_py = dist.Bernoulli(z1).log_prob(label).sum(dim=-1)
        w = torch.exp(log_qyzc_ - log_qyx)
        elbo = (w * (log_pxz - kl - log_qyzc) + log_py + log_qyx).mean()
        ######outputs from Decoder###########
        P_region,Y_region=self.decoder_net.TrainForward(x1,x2,x3,RoI,y_region)          
        return output,P_region,Y_region,RoI,LocOut,Y,-elbo
    
    def forward_RoI_Loc(self, x,y):
        LocOut,Y,x1,x2,x3,=self.encoder_net(X)
        return [LocOut,Y]    
    def forward(self,X,y,label):
        LocOut,Y,x1,x2,x3=self.encoder_net(X)    
        ######## Boundingboxes aka rois naively ###########
        RoI=self.Localization(LocOut)        
        #######################Starting on decoder side###################
        P_region=self.decoder_net(x1,x2,RoI,y_region,y_contour)
        output=self.gen_decoder(z)
        return output,P_region
    def classifier_loss(self, x, y, k=100):
        """
        Computes the classifier loss.
        """
        LocOut1=self.encoder_net.trs(x) 
        zc, _ = dist.Normal(*self.pseg_net2(LocOut1)).rsample(torch.tensor([k])).split([self.z_classify,
                                                                                self.z_style], -1)
        logits = self.classifier(zc.view(-1, self.z_classify))
        d = dist.Bernoulli(logits=logits)
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
        y=y.float()
        lqy_z = d.log_prob(y).view(k, x.shape[0], self.num_classes).sum(dim=-1)
        lqy_x = torch.logsumexp(lqy_z, dim=0) - np.log(k)
        return lqy_x
    def reconstruct_img(self, x):
        return self.decoder(dist.Normal(*self.encoder(x)).rsample())

    def classifier_acc(self, x, y=None, k=1):
        """"
        Computes accuracy for the classifier
        """"
        LocOut1=self.encoder_net.trs(x)
        zc, _ = dist.Normal(*self.pseg_net2(LocOut1)).rsample(torch.tensor([k])).split([self.z_classify,
                                                                                self.z_style], -1)
        logits = self.classifier(zc.view(-1, self.z_classify)).view(-1, self.num_classes)
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
        preds = logits
        outy=torch.argmax(y,1)
        p=torch.argmax(preds,1)
        acc = (p.eq(outy)).float().mean()
        return acc

    def accuracy(self, x,y, *args, **kwargs):
        acc = 0.0
        x, y = x.cuda(), y.cuda()
        batch_acc = self.classifier_acc(x, y)
        return batch_acc
    


