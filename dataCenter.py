import sys
import os

from collections import defaultdict
from collections import namedtuple
import numpy as np
import torch.utils.data as data
import torch
import scipy.io as sio
import h5py as hdf
from util import *
import torch.utils.data as data
import hdf5storage
import random

_paths = {
    'Flickr': '/home/jwz/gitcode/graphsage-pytorch-geometric/dataset/MIRFLICKR.mat',
    'Flickr_vigb': '/home/jwz/gitcode/graphsage-pytorch-geometric/dataset/MIRFLICKR_vigb.mat',
}

dataind = namedtuple('dataind', ['idx_train', 'idx_val', 'idx_test', 'first', 'n_t'])

def normalize(x):
    l2_norm = np.linalg.norm(x, axis=1)[:, None]
    l2_norm[np.where(l2_norm == 0)] = 1e-6
    x = x / l2_norm
    return x


def zero_mean(x, mean_val=None):
    if mean_val is None:
        mean_val = np.mean(x, axis=0)
    x -= mean_val
    return x, mean_val

def load_data(dataset):
    if dataset == 'Flickr' or dataset == 'Flickr_vigb' or dataset == 'Flickr_vit':
        data = sio.loadmat(_paths[dataset])
        features = np.float32(data['XAll'])
        labels = np.int32(data['LAll'])
        idx_train = range(18000)
        idx_val = range(18000)
        idx_test = range(18000, 20015)
        cossim, valcossim = calcos(dataset, features, idx_train, idx_val, idx_test)

        features = normalize(features)
        features, mean_val = zero_mean(features)
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels)
        
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        first = 4000
        n_t = 2000
        # edges = torch.tensor(adj_lists, dtype=torch.long)
        return features, labels, cossim, valcossim, dataind(idx_train, idx_val, idx_test, first, n_t)
    elif 'COCO' in dataset:
        # data = sio.loadmat(_paths[dataset])
        data = hdf5storage.loadmat(_paths[dataset])
        features = np.float32(data['data'])
        labels = np.int32(data['label'])
        index = [i for i in range(len(labels))] 
        random.shuffle(index)
        features = features[index]
        labels = labels[index]
        idx_train = range(40000)
        # idx_val = range(190000)
        # idx_test = range(190000, 192000)
        idx_val = range(120218)
        idx_test = range(120218, 122218)
        cossim, valcossim = calcos(dataset, features, idx_train, idx_val, idx_test)

        features = normalize(features)
        features, mean_val = zero_mean(features)
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        first = 4000
        n_t = 2000
        # edges = torch.tensor(adj_lists, dtype=torch.long)
        return features, labels, cossim, valcossim, dataind(idx_train, idx_val, idx_test, first, n_t)
    