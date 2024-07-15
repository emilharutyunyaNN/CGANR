from .layer_utils import *
from .backbone_zoo import backbone_zoo, bach_norm_checker



class UNET_left(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stack_num=2, activation='ReLU',
              pool=True, batch_norm=False):
        super(UNET_left, self).__init__()
        pool_size = 2
        if pool:
            self.enc = EncodeLayer(channel_in, channel_in, pool_size, pool, activation=activation, 
                     batch_norm=batch_norm)
        
            self.conv_st = ConvStack(channel_in, channel_out, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm)
        else:
            self.enc = EncodeLayer(channel_in, channel_out, pool_size, pool, activation=activation, 
                     batch_norm=batch_norm)
        
            self.conv_st = ConvStack(channel_out, channel_out, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm)
            
        
    def forward(self, x):
        #print((name, param.grad) for (name, param) in self.enc.)
        x = self.enc(x)
        #print("done")
        #print(x.shape)
        x = self.conv_st(x)
        return x
    
    
class UNETLeftWithRes(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stack_num=2, activation='ReLU', pool=True, batch_norm=False):
        super(UNETLeftWithRes, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stack_num = stack_num
        self.activation = getattr(nn, activation)() if activation else None
        self.pool = pool
        self.batch_norm = batch_norm
        pool_size = 2
        self.encode_layer = EncodeLayer(channel_in,channel_out, pool_size, pool, activation=activation,
                     batch_norm=batch_norm)
        self.res_conv_stack = Res_CONV_stack(channel_out,channel_out, res_num=stack_num, activation=activation,
                       batch_norm=batch_norm)
    def forward(self, x):
        x = self.encode_layer(x)
        
        # Skip connection with padding
        x_skip = F.pad(x, (0, 0, 0, 0, 0, self.channel_out - x.size(1)), "constant", 0)
        
        x = self.res_conv_stack(x,x_skip)
        
        # Add skip connection
        #x += x_skip
        
        return x
    
class UNET_right(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, 
               stack_num=2, activation='ReLU',
               unpool=True, batch_norm=False, concat=True):
        super(UNET_right, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.ks = kernel_size
        self.sn = stack_num
        self.act = activation
        self.unpool = unpool
        self.bn = batch_norm
        pool_size = 2
        self.cnct = concat
        self.dcd = DecodeLayer(channel_in,channel_in, pool_size, unpool, 
                     activation=activation, batch_norm=batch_norm)
        
        self.cnv_stk1 = ConvStack(channel_in,channel_out, kernel_size, stack_num=1, activation=activation, 
                   batch_norm=batch_norm)
        #self.cnv_stk2 = ConvStack(channel, kernel_size, stack_num=stack_num, activation=activation, 
                 #  batch_norm=batch_norm)
        
    def forward(self, x, x_list):
        #print(x.shape)
        x = self.dcd(x)
        #print(x.shape)
        x = self.cnv_stk1(x)
       # print(x.shape)
        if self.cnct:
          #  print("**")
            x = torch.cat([x,] + x_list, dim=1)
       # print(x.shape)
       # print(self.channel_out*(len(x_list)+1))
       #TODO: deal with the initialization
        cnv_final = ConvStack(self.channel_out*(len(x_list)+1), self.channel_out,  self.ks, stack_num=self.sn, activation= self.act, 
                  batch_norm=self.bn)
        x = cnv_final(x)
        return x
    
"""def test_UNETRight():
    input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels, height 32, width 32
    skip_tensor1 = torch.randn(1, 64, 32, 32)
    skip_tensor2 = torch.randn(1, 64, 32, 32)
    model = UNET_right(channel=64, kernel_size=3, stack_num=2, activation='ReLU', unpool='bilinear', batch_norm=True, concat=True)
    output_tensor = model(input_tensor, [skip_tensor1, skip_tensor2])
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

# Run the test
test_UNETRight()
"""

class unet_2d_base(nn.Module):
    def __init__(self, input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True):
        super(unet_2d_base, self).__init__()
        self.act_fn = getattr(nn, activation)()
        
        self.batch_norm = batch_norm
        self.activation = activation
        self.backbone = backbone
        self.input = input_tensor
        self.cnv_no_bck = ConvStack(input_tensor.shape[1],filter_num[0], stack_num=stack_num_down, activation=activation, 
                       batch_norm=batch_norm)
        self.filter_num = filter_num
        self.unpool = unpool
        self.stack_num_up = stack_num_up
        self.stack_num_down = stack_num_down
        self.pool = pool
        self.weights = weights
        self.fb = freeze_backbone
        self.fbn = freeze_batch_norm
    def forward(self, x):
        depth_ = len(self.filter_num)
        X_skip = []
        #print(self.backbone)
        if self.backbone is None:
            x = self.input
           # print(x.shape)
            x = self.cnv_no_bck(x)
            X_skip.append(x)
            #print(x.shape)
            for i,f in enumerate(self.filter_num[1:]):
               # print(":::",x.shape[1], f)
                #TODO: deal with the initialization
                model_unet_left = UNET_left(channel_in=x.shape[1],channel_out= f, stack_num=self.stack_num_down, activation=self.activation, pool=self.pool, 
                          batch_norm=self.batch_norm)
                x = model_unet_left(x)
                #print("---",x.shape)
                
                X_skip.append(x)
        else:
            #TODO: backbone thing
            if 'vgg' in self.backbone:
                backbone_ = backbone_zoo(self.backbone, self.weights, x.shape, depth_, self.fb, self.fbn)
                X_skip = backbone_([x,])
                depth_encode = len(X_skip)
            else:
                backbone_ = backbone_zoo(self.backbone, self.weights, x.shape, depth_-1, self.fb, self.fbn)
                X_skip = backbone_([x,])
                depth_encode = len(X_skip)+1
                
            if depth_encode< depth_:
                #print("here")
                x = X_skip[-1]
                for i in range(depth_ - depth_encode):
                    i_real = i+depth_decode
                    #TODO: deal with the initialization
                    model_left_unet = UNET_left(self.filter_num[i_real], stack_num=self.stack_num_down, activation=self.activation, pool=self.pool, 
                              batch_norm=self.batch_norm)
                    x = model_left_unet(x)
                    X_skip.append(x)
            pass
        X_skip = X_skip[::-1]
        x = X_skip[0]
        x_decode = X_skip[1:]
        depth_decode = len(x_decode)
        filter_num_decode = self.filter_num[:-1][::-1]
        #print("filter decode: ", filter_num_decode)
        
        for i in range(depth_decode):
          #  print("shape: ",x.shape)
            model_unet_right = UNET_right(x.shape[1], filter_num_decode[i], stack_num=self.stack_num_up, activation=self.activation, 
                       unpool=self.unpool, batch_norm=self.batch_norm)
            x = model_unet_right(x,[x_decode[i],])
            
        if depth_decode<depth_- 1:
          #  print("---")
            for i in range(depth_ - depth_decode - 1):
                i_real = i+depth_decode
                model_unet_right = UNET_right(filter_num_decode[i_real], stack_num=self.stack_num_up, activation=self.activation, 
                       unpool=self.unpool, batch_norm=self.batch_norm, concat=False)
                x = model_unet_right(x,None)
        return x      
    
class unet_2d(nn.Module):
    def __init__(self, input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True):
        super(unet_2d, self).__init__()
        if backbone is not None:
            bach_norm_checker(backbone,batch_norm)
        self.nl = n_labels
        self.oc = output_activation
        self.unet_model = unet_2d_base(input_size, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                     activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, 
                     backbone=backbone, weights=weights, freeze_backbone=freeze_backbone, 
                     freeze_batch_norm=freeze_backbone)
        
        #self.out = CONV_output(input_size.shape[1],n_labels=n_labels, kernel_size=1, activation=output_activation)
        
    def forward(self,x):
        x = self.unet_model(x)
        #print("before conv_out:  ",x.shape)
       #TODO: deal with the initialization
        out = CONV_output(x.shape[1],n_labels=self.nl, kernel_size=1, activation=self.oc)
        x = out(x)
        
        return x
    
    
import torch
"""
def test_UNET_left():
    print("Test UNET LEFT")
    input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, height 256, width 256
    model = UNET_left(channel_in=3,channel_out = 3 ,kernel_size=3, stack_num=2, activation='ReLU', pool=True, batch_norm=False)
    output_tensor = model(input_tensor)
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

def test_UNETLeftWithRes():
    print("Test UNET LEFT with RESs")
    input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, height 256, width 256
    model = UNETLeftWithRes(channel_in=3,channel_out = 3, kernel_size=3, stack_num=2, activation='ReLU', pool=True, batch_norm=False)
    output_tensor = model(input_tensor)
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

def test_UNET_right():
    print("test UNET RIGHT")
    input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels, height 32, width 32
    skip_tensor1 = torch.randn(1, 64, 64, 64)
    skip_tensor2 = torch.randn(1, 64, 64, 64)
    model = UNET_right(channel_in=64, channel_out = 64, kernel_size=3, stack_num=2, activation='ReLU', unpool=True, batch_norm=False, concat=True)
    output_tensor = model(input_tensor, [skip_tensor1, skip_tensor2])
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

def test_unet_2d_base():
    print("test UNET 2d BASE")
    input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, height 256, width 256
    print(input_tensor.shape[1])
    model = unet_2d_base(input_tensor, filter_num=[64, 128, 256], stack_num_down=2, stack_num_up=2,
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True)
    output_tensor = model(input_tensor)
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

def test_unet_2d():
    print("test UNET 2d")
    input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, height 256, width 256
    model = unet_2d(input_size=input_tensor, filter_num=[64, 128, 256], n_labels=10, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True)
    output_tensor = model(input_tensor)
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

# Run the tests
test_UNET_left()
test_UNETLeftWithRes()
test_UNET_right()
test_unet_2d_base()
test_unet_2d()
"""