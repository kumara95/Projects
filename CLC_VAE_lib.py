################### The architecture for Capturing label characetistics - variational autoencoder##########
############# It also contains the functions related to the network training #####################
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import torch.distributions as dist


# In[ ]:


def init_weights(m):
    if isinstance(m,nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()   
class EncoderBlock(nn.Module):
    """
    Initializes the encoder network
    """
    def __init__(self,latent_dim):
        super(EncoderBlock, self).__init__()

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
        
        self.layer=nn.Sequential(nn.Linear(160,90),
                               nn.LeakyReLU(0.1),
                               nn.Linear(90,80),
                                nn.Linear(80, 2*latent_dim))
    
        self.latent_dim=latent_dim


    def forward(self, X):
        op_en = self.encoder(X)
        openm=self.mp1(op_en)
        
        op_en2=self.encoder2(openm)
        open2m=self.mp2(op_en2)
        
        op_en3=self.encoder3(open2m)
        open3m=self.mp3(op_en3)
        
        h = torch.flatten(open3m, start_dim=1)
        s=self.layer(h)
        mu=s[:,:self.latent_dim]        
        log_var2=torch.exp(s[:,self.latent_dim:])
        return mu,log_var2
     
        
class DecoderBlock(nn.Module):
    """
    Initializes the decoder network
    """
    def __init__(self,latent_dim):
        super(DecoderBlock,self).__init__()
        self.lin=nn.Sequential(nn.Linear(latent_dim,80))
        #self.lin.apply(init_weights)
        self.decoder1= nn.Sequential(nn.ConvTranspose1d(2,8,kernel_size=2,stride=2),
                                     nn.Conv1d(8,4,kernel_size=5,stride=1,padding=2),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(4,affine=False))
        self.decoder2 = nn.Sequential(nn.ConvTranspose1d(4,4,kernel_size=5,stride=5),                                     
                                      nn.Conv1d(4,4,kernel_size=5,stride=1,padding=2),
                                      nn.Tanh(),
                                      nn.BatchNorm1d(4,affine=False))
        self.decoder3=nn.Sequential(nn.ConvTranspose1d(4,2,kernel_size=5,stride=5),
                                    nn.Conv1d(2,2,kernel_size=5,stride=1,padding=2),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(2,affine=False),
                                   nn.Conv1d(2,1,kernel_size=5,stride=1,padding=2),
                                    nn.Sigmoid())                                   
                                    
    
    def forward(self,X):        
        #upx=self.upsamp(X)
        l=self.lin(X)
        num_shape=l.shape
        dec_inp=torch.reshape(l,(num_shape[0],2,40))
        op_de=self.decoder1(dec_inp)
               
        op_de1=self.decoder2(op_de)
           
        op_de2=self.decoder3(op_de1)
       
        return op_de2


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

class CCVAE(nn.Module):
    """
    CCVAE
    """
    def __init__(self, z_dim, num_classes,y_prior):
        super(CCVAE, self).__init__()
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

        self.encoder = EncoderBlock(self.z_dim)
        self.decoder = DecoderBlock(self.z_dim)
        self.classifier = Classifier(self.num_classes)
        
        #
        self.cond_prior = CondPrior(self.num_classes)

        self.ones = self.ones.cuda()
        self.zeros = self.zeros.cuda()
        self.y_prior_params = self.y_prior_params.cuda()
        self.cuda()
        

    

    def sup(self, x, y):
        """
        Computes the forward pass throught the network and the operations
        """
        bs = x.shape[0]
        #inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        zc, zs = z.split([self.z_classify, self.z_style], 1)
        output=self.classifier(zc)
        qyzc = dist.Bernoulli(logits=output)
        log_qyzc = qyzc.log_prob(y).sum(dim=-1)
        # compute kl
        locs_p_zc, scales_p_zc = self.cond_prior(y)
        prior_params = (torch.cat([locs_p_zc, self.zeros.expand(bs, -1)], dim=1), 
                        torch.cat([scales_p_zc, self.ones.expand(bs, -1)], dim=1))
        #prior_params = (self.zeros.expand(bs, -1), self.ones.expand(bs, -1))
        kl = compute_kl(*post_params, *prior_params)

        #compute log probs for x and y
        recon = self.decoder(z)
        log_qyx = self.classifier_loss(x, y)
        log_pxz = img_log_likelihood(recon, x)

        # we only want gradients wrt to params of qyz, so stop them propogating to qzx
        log_qyzc_ = dist.Bernoulli(logits=self.classifier(zc.detach())).log_prob(y).sum(dim=-1)
        z1=self.y_prior_params.expand(bs, -1)
        log_py = dist.Bernoulli(z1).log_prob(y).sum(dim=-1)
        w = torch.exp(log_qyzc_ - log_qyx)
        elbo = (w * (log_pxz - kl - log_qyzc) + log_py + log_qyx).mean()
        return -elbo

    def classifier_loss(self, x, y, k=100):
        """
        Computes the classifier loss.
        """
        zc, _ = dist.Normal(*self.encoder(x)).rsample(torch.tensor([k])).split([self.z_classify, self.z_style], -1)
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
        """
        Computes accuracy for the classifier
        """
        zc, _ = dist.Normal(*self.encoder(x)).rsample(torch.tensor([k])).split([self.z_classify, self.z_style], -1)
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
    

