import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import random
from collections import Counter
import h5py
import json
import scipy.sparse as sp
random.seed(10)

class SceneGraphDataset(data.Dataset):
    def __init__(self):
        self.scene_graph = json.load(open('../data/graph/scene_graph_list.json', 'r'))
        self.idx2label = json.load(open('../data/graph/idx2label.json', 'r'))
        self.idx2pred = json.load(open('../data/graph/idx2pred.json', 'r'))
        duplicated_labels = {}

        for (video_id, scene_graphs) in self.scene_graph.items():
            for scene_graph in scene_graphs:
                pred_labels = scene_graph['pred_labels']
                class_labels = [a.split('-')[1] for a in pred_labels]
                
                b = dict(Counter(class_labels))
                for key, value in b.items():
                    if value > 1:
                        if key not in duplicated_labels:
                            duplicated_labels[key] = value
                        else:
                            if value > duplicated_labels[key]:
                                duplicated_labels[key] = value

        self.video_id_list = list(self.scene_graph.keys())
        self.scene_graph_list = list(self.scene_graph.values())

    def __len__(self):
        return len(self.scene_graph)

    def __getitem__(self, index):
        video_id = self.video_id_list[index]

        scene_graphs = self.scene_graph_list[index]

        max_len = 6
        sg_len = np.minimum(len(scene_graphs), max_len)
        select_idx = random.sample(range(len(scene_graphs)), sg_len)
        select_idx.sort()

        feature_bank = []
        adj_bank = []

        max_rel_num = 35
        rel_num = []

        for i in range(len(select_idx)):
            sg_idx = select_idx[i]

            pred_rels = scene_graphs[sg_idx]['pred_rels'][:max_rel_num]

            rel_obj_0 = np.array([[pred_rels[k][0], str(k)+'_'+str(pred_rels[k][1])] for k in range(len(pred_rels))])
            rel_obj_1 = np.array([[str(k)+'_'+str(pred_rels[k][1]), pred_rels[k][2]] for k in range(len(pred_rels))])
            pairs = np.concatenate((rel_obj_0, rel_obj_1))
            classes = list(set(pairs.flatten()))
            rel_num.append(len(classes))
            
            # import pdb; pdb.set_trace()

            idx_map = {j: i for i, j in enumerate(classes)}
            edges = np.array(list(map(idx_map.get, pairs.flatten())), dtype=np.int32).reshape(pairs.shape)

            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(idx_map), len(idx_map)), dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

            features = torch.zeros(adj.shape[0], len(self.idx2label)+len(self.idx2pred))
            obj_set = {}
            for rel in pairs:
                for node in rel:
                    if '-' in node:
                        obj = int(node.split('-')[1])
                        features[idx_map[node], obj-1] = 1
                    else:
                        # import pdb; pdb.set_trace()
                        rel = int(node.split('_')[1])
                        features[idx_map[node], len(self.idx2label)+rel-1] = 1


            features = sp.csr_matrix(features, dtype=np.float32)

            features = normalize_features(features)
            adj = normalize_adj(adj + sp.eye(adj.shape[0])) 
            feature_bank.append(features)
            adj_bank.append(adj)

        return video_id, feature_bank, adj_bank, rel_num

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    # import warnings
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    # with warnings.catch_warnings():
    #     import pdb; pdb.set_trace()
    #     print("1")
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def collate_fn(data):

    video_id, feature_bank, adj_bank, rel_num = zip(*data)

    return video_id, feature_bank, adj_bank, rel_num

def get_loader(batch_size, shuffle, num_workers, drop_last=False):
    dataset = SceneGraphDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                            num_workers=num_workers, collate_fn=collate_fn)
    
    return dataset, data_loader