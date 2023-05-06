import torch
import heapq
import math
import time
import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
import scipy.io as sio
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from collections import namedtuple
from torch.autograd import Variable
from model import *


def encode_onehot(labels):                                   # 把标签转换成onehot
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[
        i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def labels_affnty(labels_1, labels_2):
    if (isinstance(labels_1, torch.LongTensor) or
            isinstance(labels_1, torch.Tensor)):
        labels_1 = labels_1.numpy()
    if (isinstance(labels_2, torch.LongTensor) or
            isinstance(labels_2, torch.Tensor)):
        labels_2 = labels_2.numpy()

    if labels_1.ndim == 1:
        affnty = np.float32(labels_1[:, np.newaxis] == labels_2)
    else:
        affnty = np.float32(np.sign(np.dot(labels_1, labels_2.T)))
    return affnty

def calcos(dataset, features, idx_train, idx_val, idx_test):
    test = features[idx_test, :]
    val = features[len(idx_train):len(idx_val), :]
    train = features[:4000, :]
    norm1 = norm(test,axis=-1).reshape(test.shape[0],1)
    norm2 = norm(train,axis=-1).reshape(1,train.shape[0])
    end_norm = np.dot(norm1,norm2)
    cossim = np.dot(test, train.T) / end_norm
    I=[]
    k_cossim = np.zeros((test.shape[0], train.shape[0]), dtype = int)
    for i in range(len(cossim)):
        b=heapq.nlargest(200, range(len(cossim[i])), cossim[i].take)
        I.append(b)
    for i in range(k_cossim.shape[0]):
        k_cossim[i,I[i]] = 1;
    
    norm1 = norm(val,axis=-1).reshape(val.shape[0],1)
    end_norm = np.dot(norm1,norm2)
    valcossim = np.dot(val, train.T) / end_norm
    I=[]
    k_valcossim = np.zeros((val.shape[0], train.shape[0]), dtype = int)
    for i in range(len(valcossim)):
        b=heapq.nlargest(200, range(len(valcossim[i])), valcossim[i].take)
        I.append(b)
    for i in range(k_valcossim.shape[0]):
        k_valcossim[i,I[i]] = 1;
    return k_cossim, k_valcossim

def cross_entropy(logits, y):
    s = torch.exp(logits)
    logits = s / torch.sum(s, dim=1, keepdim=True)
    c = -(y * torch.log(logits)).sum(dim=-1)
    return torch.mean(c)

def apply_model(model, optimizer, features, labels, args, dataind, device):
    idx_train = dataind.idx_train
    trainsize = len(idx_train) + 1
    
    first = dataind.first
    n_t = dataind.n_t
    n_bits = args.n_bits
    b_sz = args.b_sz
    tH = []
    visited_nodes = set()
    for t in range(first, trainsize, n_t):
        if t == first:
            train_node = idx_train[:first]
            batches = math.ceil(len(train_node) / b_sz)

            for index in range(batches):
                nodes_batch = train_node[index * b_sz:(index + 1) * b_sz]
                visited_nodes |= set(nodes_batch)
                if labels.ndim == 1:
                    labels_batch = labels[nodes_batch]
                else:
                    labels_batch = labels[nodes_batch, :]
                adj_lists = [[], []]
                affinity = labels_affnty(labels_batch, labels_batch)
                for (row, col), val in np.ndenumerate(affinity):
                    if row != col and val != 0:
                        adj_lists[0].append(row)
                        adj_lists[1].append(col)
                        adj_lists[0].append(col)
                        adj_lists[1].append(row)
                edges = torch.tensor(adj_lists, dtype=torch.long)
                # feed nodes batch to the graphSAGE
                # returning the nodes embeddings
                optimizer.zero_grad()
                logists, embs_batch = model(
                    features[index * b_sz:(index + 1) * b_sz, :].to(device), edges.to(device))
                tH.append(embs_batch.data.cpu().numpy())

                # logloss = nn.MultiLabelSoftMarginLoss(reduction='mean')
                logloss = nn.BCELoss()
                quanloss = nn.MSELoss()

                Sim = affinity
                Sim = torch.tensor(Sim).to(device)
                B = torch.sign(embs_batch)
                binary_target = Variable(B).cuda()
                labels_batch = labels_batch.to(torch.float)
                labels_batch = labels_batch.to(device)
                loss = ((n_bits * Sim - embs_batch @ B.t())**2).mean()*args.alpha # Flickr: 0.01
                loss += quanloss(embs_batch, binary_target) * args.lambd1  # Flickr: 0.001
                loss += cross_entropy(logists, labels_batch) *args.lambd2 

                print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                                len(visited_nodes), len(idx_train)))
                loss.backward()
                optimizer.step()
        else:
            train_node = idx_train[t - n_t: t]
            label_node = labels[t - n_t: t]
            batches = math.ceil(len(train_node) / b_sz)

            for index in range(batches):
                nodes_batch = train_node[index * b_sz:(index + 1) * b_sz]
                visited_nodes |= set(nodes_batch)
                if labels.ndim == 1:
                    labels_batch = labels[nodes_batch]
                else:
                    labels_batch = labels[nodes_batch, :]
                adj_lists = [[], []]
                S = labels_affnty(labels_batch, labels_batch)
                for (row, col), val in np.ndenumerate(S):
                    if row != col and val != 0:
                        adj_lists[0].append(row)
                        adj_lists[1].append(col)
                        adj_lists[0].append(col)
                        adj_lists[1].append(row)
                edges = torch.tensor(adj_lists, dtype=torch.long)

                optimizer.zero_grad()
                logists, embs_batch = model(
                    features[nodes_batch[0]: nodes_batch[-1]+1, :].to(device), edges.to(device))
                tH.append(embs_batch.data.cpu().numpy())

                logloss = nn.BCELoss()
                quanloss = nn.MSELoss()

                old_label = labels[index * b_sz:(index + 2) * b_sz]
                oH = torch.tensor(np.reshape(np.array(tH),(-1, n_bits))).to(device)
                Sim = labels_affnty(old_label, labels_batch)
                # Sim = S
                Sim = torch.tensor(Sim).to(device)
                S = torch.tensor(S).to(device)
                B = torch.sign(embs_batch)
                binary_target = Variable(B).cuda()
                labels_batch = labels_batch.to(torch.float)
                labels_batch = labels_batch.to(device)
                loss = ((n_bits * S - embs_batch @ B.t())**2).mean()*args.alpha # Flickr: 0.01
                loss +=((n_bits * Sim - oH[index * b_sz:(index + 2) * b_sz] @ embs_batch.t())**2).mean()*args.beta # Flickr: 0.01
                loss += quanloss(embs_batch, binary_target) * args.lambd1 # Flickr: 0.001
                loss += cross_entropy(logists, labels_batch) *args.lambd2

                print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                                len(visited_nodes), len(idx_train)))
                loss.backward()
                optimizer.step()
    tH = np.concatenate(tH)
    return model, tH

def generate_code(model, cossim, valcossim, features, labels, tH, dataind, device):
    idx_train = dataind.idx_train
    idx_test = dataind.idx_test 
    train_nodes = dataind.idx_val
    model.eval()
    trainemb = []
    for t in range(len(idx_train), len(train_nodes), 1000):
        train_node = train_nodes[t:t + 1000]
        tradj_list = [[], []]
        num_node = t - len(idx_train)
        affnty = valcossim[num_node: num_node + 1000, :]
        l = affnty.shape[0]
        r = affnty.shape[1]
        for (row, col), val in np.ndenumerate(affnty):
            if val != 0:
                tradj_list[0].append(row)
                tradj_list[1].append(col + l)
                tradj_list[0].append(col + l)
                tradj_list[1].append(row)
        edges = torch.tensor(tradj_list, dtype=torch.long)
        fea = torch.cat((features[train_node[0]: train_node[-1]+1, :],features[:r, :]))
        logists, tremb = model(fea.to(device), edges.to(device))
        trainemb.append(tremb[:len(train_node), :].data.cpu().numpy())
    if len(trainemb):
        trainemb = np.concatenate(trainemb)
        trainembs = np.vstack((tH, trainemb))
    else:
        trainembs = tH
    trainembs = torch.Tensor(trainembs)
    # fea = torch.cat((features[59000:60000, :],features[:1000, :]))

    start = time.perf_counter();
    tsadj_list = [[], []]
    affnty = cossim[:, :]
    l = affnty.shape[0]
    r = affnty.shape[1]
    for (row, col), val in np.ndenumerate(affnty):
        if val != 0:
            tsadj_list[0].append(row)
            tsadj_list[1].append(col + l)
            tsadj_list[0].append(col + l)
            tsadj_list[1].append(row)
    tsadj_list = torch.tensor(tsadj_list, dtype=torch.long)
    fea = torch.cat((features[idx_test, :],features[:r, :]))
    logists, testembs = model(fea.to(device), tsadj_list.to(device))
    print('retTime:', time.perf_counter() - start)

    tst_label = labels[idx_test]
    trn_label = labels[train_nodes]
    tst_binary = torch.sign(testembs[:len(idx_test), :]).to(device)
    trn_binary = torch.sign(trainembs).to(device)
    return trn_binary, tst_binary, trn_label, tst_label

def compute_mAP(trn_binary, tst_binary, trn_label, tst_label, n_bits):
    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum(
            (query_binary != trn_binary).long(), dim=1).sort()

        correct = torch.as_tensor((query_label == trn_label[query_result])).float()
        P = torch.cumsum(correct, dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    mAP = torch.mean(torch.Tensor(AP))
    print('%d bits: mAP: %.4f' % (n_bits, mAP))
    return mAP

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def CalcTopMap(rB, qB, retrieval_L, query_L, n_bits, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)      # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    mAP = map / num_query
    print('%d bits: mAP: %.4f' % (n_bits, mAP))
    return map