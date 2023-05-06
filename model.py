import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.datasets import Planetoid

class Classification(nn.Module):
	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)
							)
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.log_softmax(self.layer(embeds), 1)
		return logists

class GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden, out_size, num_classes, n_bits):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)
        self.sage2 = SAGEConv(hidden, out_size)
        self.conv = nn.Linear(out_size, n_bits)
        self.clas = nn.Linear(out_size, num_classes)
        self.BN = nn.BatchNorm1d(n_bits)
        self.act = nn.Tanh()

    def forward(self, features, edges):
        features = self.sage1(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.sage2(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)  # 增加了dropout
        # logists = torch.log_softmax(self.clas(features), 1)
        # logists = torch.sigmoid(self.clas(features))
        logists = self.clas(features)
        pre_hidden_embs = self.conv(features)
        pre_hidden_embs = self.BN(pre_hidden_embs)
        out = self.act(pre_hidden_embs)
        return logists, out