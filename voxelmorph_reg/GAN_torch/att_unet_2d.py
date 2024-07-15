from .layer_utils import *
from .activations import GELU, Snake
from .unet_2d import UNET_left, UNET_right
from .backbone_zoo import backbone_zoo, bach_norm_checker
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UNET_att_right(nn.Module):
    def __init__(self,channel_in, channel_out, att_channel_in, att_channel_out,rank, kernel_size=3, stack_num=2,
                   activation='ReLU', atten_activation='ReLU', attention='add',
                   unpool=True, batch_norm=False):
        
        """ print("UNET RIGHT PARAMS")
        print("kernel size: ", kernel_size)
        print("stack number: ", stack_num)
        print("activation: ", activation)
        print("attention activation: ", atten_activation )
        print("attention: ", attention)
        print("unpool: ", unpool)
        print("batch norm: ", batch_norm)"""
        super(UNET_att_right, self).__init__()
        pool_size = 2
        self.kernel_size = kernel_size
        self.stack_num = stack_num
        self.activation = activation
        self.batch_norm = batch_norm
        self.dec = DecodeLayer(channel_in, channel_out, pool_size, unpool, 
                     activation=activation, batch_norm=batch_norm).to(torch.device(rank))
        self.left = AttentionGate(channel_in=att_channel_in, channel_in_g=channel_in,channel_out=att_channel_out,activation=atten_activation, 
                            attention=attention).to(torch.device(rank))
        if unpool:
            self.cnv_stk = ConvStack(channel_in=channel_in+att_channel_in,channel_out=channel_out, kernel_size=kernel_size, stack_num=stack_num, activation=activation, 
                  batch_norm=batch_norm).to(torch.device(rank))
        else:
            self.cnv_stk = ConvStack(channel_in=channel_out+att_channel_in,channel_out=channel_out, kernel_size=kernel_size, stack_num=stack_num, activation=activation, 
                  batch_norm=batch_norm).to(torch.device(rank))
        self.channel_out = channel_out
    def forward(self,x,x_left):
        #print("INSIDE RIGHT ...")
       # print(x.shape)
        x1 = self.dec(x)
       # print("decoded x : ", x1.shape)
        #print("x1 in unet att right: ", x1.shape)
        #del x
        #print(x.shape)
        #print(x_left.shape)
        #print("x left in unet att right: ",x_left.shape)
        x_left = self.left(x =x_left,g =x1)
       # print(x_left.shape)
        h = torch.cat([x1,x_left], dim = 1)
        #print("conv_stack: ", x1.shape, x_left.shape, h.shape)
        #cnv_stk = ConvStack(h.shape[1],self.channel_out, self.kernel_size, stack_num=self.stack_num, activation=self.activation, 
             #    batch_norm=self.batch_norm).to(x1.device)
        
        h1 = self.cnv_stk(h)
        #del h
        #del cnv_stk
        #torch.cuda.empty_cache()
        return h1

class att_unet_2d_base(nn.Module):
    def __init__(self,input_tensor, filter_num,rank, stack_num_down=2, stack_num_up=2,
                     activation='ReLU', atten_activation='ReLU', attention='add', batch_norm=False, pool=True, unpool=True, 
                     backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True):
        
        super(att_unet_2d_base,self).__init__()
        self.act_fn = getattr(nn, activation)()
        print("----->", input_tensor, filter_num,rank, stack_num_down, stack_num_up,
                     activation, atten_activation, attention, batch_norm, pool, unpool)
        self.attention = attention
        self.atten_activation = atten_activation
        self.batch_norm = batch_norm
        self.activation = activation
        self.backbone = backbone
        self.input = input_tensor
        self.cnv_no_bck = ConvStack(input_tensor[0],filter_num[0], stack_num=stack_num_down, activation=activation, 
                       batch_norm=batch_norm).to(torch.device(rank))
        self.left_list = nn.ModuleList([UNET_left(filter_num[i-1],filter_num[i], stack_num=stack_num_down, activation=activation, pool=pool, 
                          batch_norm=batch_norm).to(torch.device(rank)) for i in range(1,len(filter_num))])
        rev_filter = filter_num[::-1]
        
        self.right_list =nn.ModuleList([UNET_att_right(rev_filter[i-1],rev_filter[i], rev_filter[i],att_channel_out=rev_filter[i]//2,rank =rank,kernel_size=3,stack_num=stack_num_up,
                        activation=activation, atten_activation=atten_activation, attention=attention,
                        unpool=unpool, batch_norm=batch_norm).to(torch.device(rank)) for i in range(1, len(rev_filter))])
        
        self.filter_num = filter_num
        self.unpool = unpool
        self.stack_num_up = stack_num_up
        self.stack_num_down = stack_num_down
        self.pool = pool
        self.weights = weights
        self.fb = freeze_backbone
        self.fbn = freeze_batch_norm
        self.rank = rank
    def forward(self, x):
        #print("input to unet: ", x.shape)
        depth_ = len(self.filter_num)
        X_skip = []
        #print("-------Weights------ ", "\n")
       # print([(name, param.grad) for (name, param) in self.cnv_no_bck.named_parameters()])
        #print("------- ATTENTION RIGHT -------")
        #right = [b.named_parameters() for b in self.right_list]
        #for r in right:
         #   print([(name, param.grad) for (name, param) in  r] )
          #  break
        #print("------- LEFT -------")
       # left = [b.named_parameters() for b in self.left_list]
        #for l in left:
        #    print([(name, param.grad) for (name, param) in  l] )
        if self.backbone is None:
            #print("here")
            #print("---------- INSIDE G UNET ------- ")
            #x = self.input
            x = self.cnv_no_bck(x)
            #print("First convolution: ", x.shape)
            X_skip.append(x)
            #del x1
            #print("ENCODING ...")
            for i,f in enumerate(self.filter_num[1:]):
                #print("shape before left: ", X_skip[-1].shape)
                #model_unet_left = UNET_left(X_skip[-1].shape[1],f, stack_num=self.stack_num_down, activation=self.activation, pool=self.pool, 
                 #         batch_norm=self.batch_norm).to(torch.device(self.rank))
                x = self.left_list[i](X_skip[-1])
                X_skip.append(x)
                #del x
               # del model_unet_left
                #torch.cuda.empty_cache()
            #print("ENCODED x shape: ", X_skip[-1].shape)
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
                #print("in depth left <-----")
                x = X_skip[-1]
                for i in range(depth_ - depth_encode):
                    i_real = i+depth_encode
                    model_left_unet = UNET_left(x.shape[1],self.filter_num[i_real], stack_num=self.stack_num_down, activation=self.activation, pool=self.pool, 
                              batch_norm=self.batch_norm).to(torch.device(self.rank))
                    x = model_left_unet(x)
                    X_skip.append(x)
        X_skip = X_skip[::-1]
        # upsampling begins at the deepest available tensor
        X = X_skip[0]
        # other tensors are preserved for concatenation
        X_decode = X_skip[1:]
        #print("before: ", torch.cuda.memory_allocated())

        #del X_skip
        #print("after: ", torch.cuda.memory_allocated())
       # print(X_skip)
        depth_decode = len(X_decode)
        #print("DECODING ...")
        # reverse indexing filter numbers
        filter_num_decode = self.filter_num[:-1][::-1]
        for i in range(depth_decode):
            #print("X shape: " ,X.shape)
            #print("filter: ", filter_num_decode[i])
            f = filter_num_decode[i]
            #print("X and X_decode before decoder: ", X.shape, X_decode[i].shape)
           # model_unet_att_right = UNET_att_right(X.shape[1],f, f,att_channel_out=f//2,rank =self.rank,kernel_size=3,stack_num=self.stack_num_up,
             #           activation=self.activation, atten_activation=self.atten_activation, attention=self.attention,
                 #       unpool=self.unpool, batch_norm=self.batch_norm).to(torch.device(self.rank))
            #print("----: ", X.shape, X_decode[i].shape, f)
            X = self.right_list[i](X, X_decode[i])
            # del model_unet_att_right
           # torch.cuda.empty_cache()
        #del X_decode
        if depth_decode < depth_-1:
            #print("in depth ****")
            for i in range(depth_-depth_decode-1):
                
                i_real = i + depth_decode
                model_right = UNET_right(X.shape[1],filter_num_decode[i_real], stack_num=self.stack_num_up, activation=self.activation, 
                    unpool=self.unpool, batch_norm=self.batch_norm, concat=False).to(torch.device(self.rank))
                X = model_right(X, None)
                #del model_right
               # torch.cuda.empty_cache()
                
        return X

class att_unet_2d(nn.Module):
    def __init__(self,input_size, filter_num, n_labels, rank,stack_num_down=2, stack_num_up=2, activation='ReLU', 
                atten_activation='ReLU', attention='add', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
                backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True):
        super(att_unet_2d, self).__init__()
        self.act_fn = getattr(nn, activation)
        if backbone is not None:
            bach_norm_checker(backbone, batch_norm)
            
        self.dsc = att_unet_2d_base(input_size, filter_num, rank,stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                         activation=activation, atten_activation=atten_activation, attention=attention,
                         batch_norm=batch_norm, pool=pool, unpool=unpool, 
                         backbone=backbone, weights=weights, freeze_backbone=freeze_backbone, 
                         freeze_batch_norm=freeze_backbone).to(torch.device(rank))
        self.conv_out = CONV_output(filter_num[0], n_labels, kernel_size=1, activation=output_activation).to(torch.device(rank))
        self.n_labels = n_labels
        self.output_activation = output_activation
        self.rank = rank
    def forward(self, x):
        x = self.dsc(x)
        #print("x1 after unet part, before conv out: ", x.shape)
        #conv_out = CONV_output(in_channels=x1.shape[1], n_labels=self.n_labels, kernel_size=1, activation=self.output_activation).to(x1.device)
        x = self.conv_out(x)
        #print("Generator output shape: ", x.shape)
        #torch.cuda.empty_cache()
        return x
