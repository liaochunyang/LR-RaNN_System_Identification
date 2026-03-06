#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 20:19:59 2025

@author: liaochunyang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random



############## Lipschitz Regularized Neural Network: https://github.com/enegrini/System-identification-through-Lipschitz-regularized-neural-networks
def Quot(input1, input2, model):
    model1 = model(input1)
    model2 = model(input2)
    Num = torch.norm(model1-model2,2)
    Den = torch.norm(input1-input2, 2)
    div = torch.div(Num,Den)
    return div


def Lip(input_matrix, how_many_points, model):
    '''
    select how_many_points random points in the input and compute and approximate Lipschitz constant with them.
    '''
    which_points = random.sample(range(0, len(input_matrix)), how_many_points)    
    Quot_vec = torch.Tensor().reshape(-1,1)
    for in1 in range(len(which_points)):
        for in2 in range(in1+1, len(which_points)-1):
            first_pt = input_matrix[which_points[in1]]
            second_pt = input_matrix[which_points[in2]]
            Quot_vec = torch.cat((Quot_vec,Quot(first_pt, second_pt, model).reshape(-1,1)))
    maxim =(torch.max(Quot_vec)).reshape(-1,1)
    return maxim

class LipNet(torch.nn.Module):
    """Lipschitz Regularized Network, 8 layers, LeakyReLU activation,
    input size is number of columns in input matrix, here is 2 for (t,x)
    output size is 1, value of RHS at given couple (t,x), hidden_size1, hidden_size2 can be chosen by user"""
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(LipNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_size,   hidden_size1, bias=True)
        self.linear2 = torch.nn.Linear(hidden_size1, hidden_size2, bias=True)
        self.linear3 = torch.nn.Linear(hidden_size2, hidden_size2, bias=True)
        self.linear4 = torch.nn.Linear(hidden_size2, hidden_size2, bias=True)
        self.linear5 = torch.nn.Linear(hidden_size2, hidden_size2, bias=True)
        self.linear6 = torch.nn.Linear(hidden_size2, hidden_size2, bias=True)
        self.linear7 = torch.nn.Linear(hidden_size2, hidden_size2, bias=True)
        self.linear8 = torch.nn.Linear(hidden_size2, output_size,  bias=True)
        
        self.lrelu = torch.nn.LeakyReLU()
        
    def forward(self, x):
        x = self.lrelu(self.linear1(x))
        x = self.lrelu(self.linear2(x))
        x = self.lrelu(self.linear3(x))
        x = self.lrelu(self.linear4(x))
        x = self.lrelu(self.linear5(x))
        x = self.lrelu(self.linear6(x))
        x = self.lrelu(self.linear7(x))
        x = self.linear8(x)
        return x




################# Lipschitz Neural Network
## from https://github.com/locuslab/orthogonal-convolutions
def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape 
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

def get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return torch.sigmoid
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'identity':
        return lambda x: x
    else:
        raise ValueError("Unsupported activation function: {}".format(activation))

## from https://github.com/acfr/LBDN
class SandwichFc(nn.Linear): 
    def __init__(self, in_features, out_features, bias=True, activation=F.relu, scale=1.0):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm() 
        self.scale = scale 
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))   
        self.Q = None
        self.activation = activation
    
    def forward(self, x): # Eq. (9)
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach() # Q = [A^T, B^T]
        x = F.linear(self.scale * x, Q[:, fout:]) # B*h 
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5) # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = self.activation(x) * torch.exp(self.psi) # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T) # sqrt(2) A^top \Psi z
        return x
    
class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale   
        self.AB = AB
        self.Q = None

    def forward(self, x): # Eq. (9)
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:]) # B @ x 
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T) # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x

class LipNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, gamma, scale=1):
        super(LipNN, self).__init__()
        self.gamma = gamma
        self.layer1 = SandwichFc(input_size, hidden_size1, bias=True, activation=F.relu, scale=scale)
        self.layer2 = SandwichFc(hidden_size1, hidden_size2, bias=True, activation=F.relu, scale=scale)
        self.layer3 = SandwichFc(hidden_size2, hidden_size2, bias=True, activation=F.relu, scale=scale)
        self.layer4 = SandwichFc(hidden_size2, hidden_size2, bias=True, activation=F.relu, scale=scale)
        self.layer5 = SandwichFc(hidden_size2, hidden_size2, bias=True, activation=F.relu, scale=scale)
        self.layer6 = SandwichFc(hidden_size2, hidden_size2, bias=True, activation=F.relu, scale=scale)
        self.layer7 = SandwichFc(hidden_size2, hidden_size2, bias=True, activation=F.relu, scale=scale)
        self.layer8 = SandwichLin(hidden_size2, output_size, bias=True, scale=self.gamma, AB=False)
        
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.layer8(x)
        return x