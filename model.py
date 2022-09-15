import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self, input_dims, hidden_dims=100):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims

        self.Q = nn.Linear(self.input_dims, self.hidden_dims, grad=True)
        self.K = nn.Linear(self.input_dims, self.hidden_dims)
        self.V = nn.Linear(self.input_dims, self.hidden_dims)

        self.softmax = nn.Softmax()

    def forward(self, X, Z=None):

        if not Z:
            Z = X

        Q_ = self.Q(X)
        K_ = self.K(Z)
        V_ = self.V(Z)

        score = (K_.T*Q_)/self.hidden_dims
        return V_*self.softmax(score)


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, attn=False):

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn = attn

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.batchnorm = nn.BatchNorm1d(self.out_dim)
        self.dropout = nn.Dropout(p=0.2)

        if self.attn:
            self.attention = Attention(self.out_dim, self.out_dim)
            self.layernorm = nn.LayerNorm(self.out_dim)


    def forward(self, X, condition=None):
        if condition:
            X = X + condition
        if self.attn:
            Z = self.lrelu(self.batchnorm(self.linear(X)))
            Z = Z + self.attention(Z)
            return self.dropout(self.layernorm(Z))
        return self.dropout(self.lrelu(self.batchnorm(self.linear(X))))


class UNet(nn.Module):

    def __init__(self, nodes: list):
        super().__init__()
        self.nodes = nodes
        self.squeeze = [Block(self.nodes[i], self.nodes[i+1], attn=True) for i in range(len(self.nodes)-1)]
        self.unsqueeze = [Block(self.nodes[i], self.nodes[i-1], attn=True) for i in range(len(self.nodes)-1, 0, -1)]

    def forward(self, X, conditon=None):

        if condition:
            X = X + condition

        squeeze_latents = [X]
        for b in self.squeeze:
            X = b(X)
            squeeze_latents.append(X)

        for i,b in enumerate(self.unsqueeze):
            X = b(X) + squeeze_latents[len(squeeze_latents)-1 - i]

        return X
