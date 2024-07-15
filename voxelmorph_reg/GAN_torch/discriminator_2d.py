from .backbone_zoo import backbone_zoo, bach_norm_checker
from .layer_utils import *
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class UNET_left(nn.Module):
    def __init__(self, channel_in, channel_out, rank, kernel_size=3, stack_num=2, activation='ReLU',
              pool=True, batch_norm=False):
        super(UNET_left, self).__init__()
        pool_size = 2
        self.enc = EncodeLayer(channel_in, channel_out, pool_size, pool, rank=rank,kernel_size='auto',activation=activation, 
                     batch_norm=batch_norm).to(torch.device(rank))
        
        self.conv_st = ConvStack(channel_out,channel_out, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm).to(torch.device(rank))
        self.rank = rank
    def forward(self, x):
        #print(f"forward with rank {self.rank}-------------", str(x.device), str(self.rank))
        if str(self.rank) not in str(x.device):
            #print("**")
            x =x.to(torch.device(self.rank))
        #print(x.device)
        x = self.enc(x)
        #print("D left x after encoder: ", x.shape)
        #del x
        #torch.cuda.empty_cache()
        #print("done")
       # print(x.shape)
        x = self.conv_st(x)
        #print("D left x after conv: ", x.shape)
        #del x1
        #torch.cuda.empty_cache()
        #x = x.to
        return x
    
    
class discriminator_base(nn.Module):
    def __init__(self,input_size, filter_num, rank=0, stack_num_down=2,
                       activation='ReLU', batch_norm=False, pool=True,
                       backbone=None):
        super(discriminator_base, self).__init__()
        self.conv = ConvStack(input_size[0],filter_num[0], stack_num=stack_num_down, activation=activation,
                   batch_norm=False).to(torch.device(rank))
        self.conv_left = nn.ModuleList([UNET_left(filter_num[i-1],filter_num[i], rank, kernel_size=3,stack_num=stack_num_down, activation=activation, pool=False,
                        batch_norm=False).to(torch.device(rank)) for i in range(1, len(filter_num))])
        #self.avg = nn.AdaptiveAvgPool2d((1,1)).to(torch.device(rank))
        self.final_dense = DenseLayer(input_features=filter_num[-1], units=1, activation=None).to(torch.device(rank))
        self.inter_dense = DenseLayer(input_features=filter_num[-1], units=filter_num[-1], activation='LeakyReLU').to(torch.device(rank))
        self.filter_num = filter_num
        self.stack_num_down = stack_num_down
        self.activation = activation
        self.rank = rank
    def forward(self, x):
        #print("initial discriminator input: ", x.shape)
        X_skip = []
        x = self.conv(x)
        #print("conv res: ", x.shape)
        X_skip.append(x)
        for i,f in enumerate(self.filter_num[1:]):
            #print(x.shape, " -----> ", i, " : ", f)
            #print("In D left before: ", x.shape)
            x = self.conv_left[i](x)
            #sdel unet_left
            #print("D after left: ", x.shape)
           # torch.cuda.empty_cache()
            
        #print("after left: ", x.shape)
        x = torch.mean(x, dim=(2, 3))
        #print("after avg: ",x.shape)
        ch = x.view(x.size(0), -1)
       # print("CH shape----------------------: ",ch.shape)
        x = self.inter_dense(ch)  # First dense layer
        x = self.final_dense(x)  # LeakyReLU activation
        #x = self.dense_layer_2(x)  # Second dense layer
        x = torch.sigmoid(x)
        #print("here")
        return x
# this one is a little sus
class discriminator_2d(nn.Module):
    def __init__(self,input_size, filter_num, rank = 1, stack_num_down=2,
                     activation='ReLU', batch_norm=False, pool=False,
                     backbone=None):
        super(discriminator_2d, self).__init__()
        
        #self.act_fn = getattr(nn, activation)
        if backbone is not None:
            bach_norm_checker(backbone, batch_norm)
            
        self.dsc = discriminator_base(input_size = input_size, filter_num=filter_num, rank = rank, stack_num_down=stack_num_down,
                           activation=activation, batch_norm=batch_norm, pool=pool).to(torch.device(rank))
        
    def forward(self, x):
        x = self.dsc(x)
        return x
    
import torch
import torch.nn as nn




# Assuming the necessary classes and functions are imported or defined in your script.
# Since the imports are specific to your environment, they are assumed to be correct here.

