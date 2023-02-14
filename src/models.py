# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:57:01 2023

@author: Iris Vos
@email: i.n.vos-6@umcutrecht.nl

"""

import torch
import torch.nn.functional as F
import torch_geometric.nn as geom_nn

class GCN(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, conv_operator):
        super().__init__()
        
        self.conv_operator = conv_operator
        self.conv1 = getattr(geom_nn, conv_operator)(in_channels, hidden_channels)
        self.conv2 = getattr(geom_nn, conv_operator)(hidden_channels, out_channels)
        

    def forward(self, data, mcdropout=False):
        self.mcdropout = mcdropout
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr  
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        
        if self.mcdropout:
            x = F.dropout(x, p=0.5, training=True)

        x = self.conv2(x, edge_index)
        
        return x


    
def reset_model_weights(model):

  for layer in model.children():
      
   if hasattr(layer, 'reset_parameters'):
    layer.reset_parameters()