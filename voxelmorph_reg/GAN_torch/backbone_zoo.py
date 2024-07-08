import torch
import torch.nn as nn
import torchvision.models as models
import warnings

def freeze_model(model, freeze_batch_norm=False):
    for name, param in model.named_parameters():
        if not freeze_batch_norm and 'bn' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model
layer_candidates = {
    'vgg16': ['features.3', 'features.8', 'features.15', 'features.22', 'features.29'],
    'vgg19': ['features.3', 'features.8', 'features.17', 'features.26', 'features.35'],
    'resnet50': ['relu', 'layer1', 'layer2', 'layer3', 'layer4'],
    'resnet101': ['relu', 'layer1', 'layer2', 'layer3', 'layer4'],
    'resnet152': ['relu', 'layer1', 'layer2', 'layer3', 'layer4'],
    'resnet50_v2': ['conv1', 'layer1', 'layer2', 'layer3', 'layer4'],
    'resnet101_v2': ['conv1', 'layer1', 'layer2', 'layer3', 'layer4'],
    'resnet152_v2': ['conv1', 'layer1', 'layer2', 'layer3', 'layer4'],
    'densenet121': ['features.denseblock1', 'features.denseblock2', 'features.denseblock3', 'features.denseblock4', 'features.norm5'],
    'densenet169': ['features.denseblock1', 'features.denseblock2', 'features.denseblock3', 'features.denseblock4', 'features.norm5'],
    'densenet201': ['features.denseblock1', 'features.denseblock2', 'features.denseblock3', 'features.denseblock4', 'features.norm5'],
    'efficientnet_b0': ['features.2', 'features.3', 'features.4', 'features.6', 'features.8'],
    'efficientnet_b1': ['features.2', 'features.3', 'features.4', 'features.6', 'features.8'],
    'efficientnet_b2': ['features.2', 'features.3', 'features.4', 'features.6', 'features.8'],
    'efficientnet_b3': ['features.2', 'features.3', 'features.4', 'features.6', 'features.8'],
    'efficientnet_b4': ['features.2', 'features.3', 'features.4', 'features.6', 'features.8'],
    'efficientnet_b5': ['features.2', 'features.3', 'features.4', 'features.6', 'features.8'],
    'efficientnet_b6': ['features.2', 'features.3', 'features.4', 'features.6', 'features.8'],
    'efficientnet_b7': ['features.2', 'features.3', 'features.4', 'features.6', 'features.8']
}

def backbone_zoo(backbone_name, weights, input_tensor_shape, depth, freeze_backbone, freeze_batch_norm):
    """
    Configuring a user-specified encoder model based on the torchvision.models

    Input
    ----------
        backbone_name: the backbone model name. Expected as one of the torchvision.models class.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0,7]

        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        input_tensor_shape: the input tensor shape as (batch_size, channels, height, width)
        depth: number of encoded feature maps. 
               If four downsampling levels are needed, then depth=4.

        freeze_backbone: True for a frozen backbone
        freeze_batch_norm: False for not freezing batch normalization layers.

    Output
    ----------
        model: a PyTorch backbone model.
        layers: list of layers for skip connections.
    """
    
    candidates = layer_candidates[backbone_name]
    
    # Depth checking
    depth_max = len(candidates)
    if depth > depth_max:
        depth = depth_max

    # Load the backbone model
    if weights == 'imagenet':
        model_func = getattr(models, backbone_name)
        model = model_func(pretrained=True)
    else:
        model_func = getattr(models, backbone_name)
        model = model_func(pretrained=False)

    if freeze_backbone:
        model = freeze_model(model, freeze_batch_norm=freeze_batch_norm)

    layers = candidates[:depth]
    return model, layers

"""# Example usage
input_tensor_shape = (8, 3, 224, 224)  # Batch of 8, 3 channels, 224x224 spatial dimensions
backbone_name = 'resnet50'
depth = 4
weights = 'imagenet'
freeze_backbone = True
freeze_batch_norm = False

model, layers = backbone_zoo(backbone_name, weights, input_tensor_shape, depth, freeze_backbone, freeze_batch_norm)
print("Backbone model layers for skip connections: ", layers)"""
def bach_norm_checker(backbone_name, batch_norm):
    if 'vgg' in backbone_name:
        batch_norm_backbone = False
    else:
        batch_norm_backbone = True
        
    if batch_norm_backbone != batch_norm:       
        if batch_norm_backbone:    
            param_mismatch = f"\n\nBackbone {backbone_name} uses batch norm, but other layers received batch_norm={batch_norm}"
        else:
            param_mismatch = f"\n\nBackbone {backbone_name} does not use batch norm, but other layers received batch_norm={batch_norm}"
            
        warnings.warn(param_mismatch)