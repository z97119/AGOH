import torch
import argparse
import numpy as np
import time
import math
import scipy.sparse as sp
import torch.nn.functional as F
from model import *
from util import *
from dataCenter import *
import scipy.io as sio

parser = argparse.ArgumentParser(description='GraphSAGE online hashing')

parser.add_argument('--dataSet', type=str, default='Flickr_vigb')
parser.add_argument('--epochs', type=int, default=10) # Flickr: 10
parser.add_argument('--b_sz', type=int, default=50)  # Flickr:50
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--n_bits', type=int, default=32)
parser.add_argument('--hid_sz', type=int, default=128)  # Flickr or coco: 128 nuswide: 512
parser.add_argument('--out_sz', type=int, default=128) # Flickr or coco: 128  nuswide: 256
parser.add_argument('--alpha', type=float, default=0.01) #Flickr: 0.01
parser.add_argument('--beta', type=float, default=0.01) #Flickr: 0.01
parser.add_argument('--lambd1', type=float, default=0.001) # Flickr: 0.001 # nuswide:0.0001 coco:0.0001
parser.add_argument('--lambd2', type=float, default=0.1) # Flickr: 0.1 # nuswide:0.01
parser.add_argument('--lr', type=float, default=0.001)  # Flickr:0.001 nuswide:0.00001  coco:0.001
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:', device)

features, labels, cossim, valcossim, dataind = load_data(args.dataSet)

start = time.perf_counter()
model = GraphSAGE(features.size(1), args.hid_sz, args.out_sz, labels.size(1), args.n_bits).to(device)
# classification = Classification(args.out_sz, labels.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# optimizer1 = torch.optim.Adam(classification.parameters(), lr=args.lr)
model.train()
# classification.train()
for epoch in range(args.epochs):
    print('----------------------EPOCH %d-----------------------' % epoch)
    model, tH = apply_model(model, optimizer, features, labels, args, dataind, device)
end = (time.perf_counter() - start)
print('trainTime:', end)

start = time.perf_counter()
trn_binary, tst_binary, trn_label, tst_label = generate_code(model, cossim, valcossim, features, labels, tH, dataind, device)
print('updateTime:', time.perf_counter()-start)

start = time.perf_counter()
if trn_label.ndim == 1:
    mAP = compute_mAP(trn_binary, tst_binary, trn_label, tst_label, args.n_bits)
else:
    mAP = CalcTopMap(trn_binary, tst_binary, trn_label, tst_label, args.n_bits)
print('retTime:', time.perf_counter()-start)

# for epoch in range(10):
#     optimizer.zero_grad()
#     out = model(features.to(device), edges.to(device))    # edges  int64
#     loss = F.nll_loss(out[idx_train], labels[idx_train].to(device))
#     loss.backward()
#     optimizer.step()
#     print(f"epoch:{epoch+1}, loss:{loss.item()}")

# _, pred = model(features.to(device), edges.to(device)).max(dim=1)
# correct = pred[idx_test].eq(labels[idx_test].to(device)).sum()
# acc = int(correct) / int(len(idx_test))
# print(acc)
