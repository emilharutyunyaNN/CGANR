import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    '''
    Gaussian Error Linear Unit (GELU), an alternative to ReLU
    
    Y = GELU()(X)
    
    ----------
    Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.
    
    Usage: use it as a nn.Module
    '''
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, X):
        return 0.5 * X * (1.0 + torch.tanh(0.7978845608028654 * (X + 0.044715 * torch.pow(X, 3))))

class Snake(nn.Module):
    '''
    Snake activation function X + (1/b)*sin^2(b*X). Proposed to learn periodic targets.
    
    Y = Snake(beta=0.5)(X)
    
    ----------
    Ziyin, L., Hartwig, T. and Ueda, M., 2020. Neural networks fail to learn periodic functions 
    and how to fix it. arXiv preprint arXiv:2006.08195.
    '''
    def __init__(self, beta=0.5, trainable=False):
        super(Snake, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=trainable)

    def forward(self, X):
        return X + (1 / self.beta) * torch.square(torch.sin(self.beta * X))

# Testing the activation functions

# Test example for GELU
"""x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], requires_grad=True)
gelu = GELU()
gelu_output = gelu(x)
print("GELU output:")
print(gelu_output)

# Test example for Snake
snake = Snake(beta=0.5, trainable=True)
snake_output = snake(x)
print("\nSnake output:")
print(snake_output)
"""