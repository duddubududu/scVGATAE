import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    """GAT layer."""
    def __init__(self,input_feature,output_feature,dropout,alpha,concat=True):
        super(GraphAttentionLayer,self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2*output_feature,1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature,output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data,gain=1.414)
        nn.init.xavier_uniform_(self.a.data,gain=1.414)
    
    def forward(self,h,adj):
        Wh = torch.mm(h,self.w)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        adj = adj.to_dense() 
        attention = torch.where(adj > 0, e, zero_vec) 
        attention = F.softmax(attention, dim=1) 
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh) 
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    def _prepare_attentional_mechanism_input(self,Wh):
        
        Wh1 = torch.matmul(Wh,self.a[:self.output_feature,:]) 
        
        Wh2 = torch.matmul(Wh,self.a[self.output_feature:,:]) 
        
        e = Wh1+Wh2.T 
        return self.leakyrelu(e)
    
