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
from collections import Counter

random.seed(10)

class SceneGraphDataset(data.Dataset):
    def __init__(self):
        self.scene_graph = json.load(open('../data/graph/scene_graph_list.json', 'r'))
        self.idx2label = json.load(open('../data/graph/idx2label.json', 'r'))
        self.idx2pred = json.load(open('../data/graph/idx2pred.json', 'r'))
        duplicated_labels = {}
        
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

        all_rels = []
        rel_num = []

        obj_count = Counter({})
        label_rel_box = []
        max_rel_num = 10

        for i in range(len(select_idx)):
            sg_idx = select_idx[i]

            pred_rels = scene_graphs[sg_idx]['pred_rels'][:max_rel_num]
            pred_labels = scene_graphs[sg_idx]['pred_labels']
            pred_labels_ = set([a for a in pred_labels for b in pred_rels if a in b])
            pred_labels = list(pred_labels_)
            boxes = scene_graphs[sg_idx]['boxes']

            pred_labels_conv = []
            for j in range(len(pred_labels)):
                obj = pred_labels[j].split('-')[1]
                if obj not in obj_count:
                    obj_count[obj] = 1
                else:
                    obj_count[obj] += 1
                conv_label = '-'.join((str(obj_count[obj]), obj))
                pred_labels_conv.append(conv_label)
                for k in range(len(pred_rels)):
                    pred_rels[k] = [conv_label if a==pred_labels[j] else a for a in pred_rels[k]]
            

            all_rels.extend(pred_rels)
            label_rel_box.append({})

            for j in range(len(pred_labels_conv)):
                obj = pred_labels_conv[j].split('-')[1]
                box = boxes[j]
                x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                center = np.array(((x1+x2)/2, (y1+y2)/2))
                if obj not in label_rel_box[i]:
                    index = 0
                    label_rel_box[i][obj] = []
                else:
                    index = len(label_rel_box[i][obj])
                label_rel_box[i][obj].append({})
                label_rel_box[i][obj][index]['coord'] = center
                label_rel_box[i][obj][index]['label'] = pred_labels_conv[j]
                rel = [r for r in pred_rels if pred_labels_conv[j] in r]
                label_rel_box[i][obj][index]['rel'] = rel

                if i != 0:
                    if obj in label_rel_box[i-1]:
                        for a in label_rel_box[i-1][obj]:
                            dist = np.linalg.norm(a['coord'] - center)
                            if dist <= 150:
                                label_rel_box[i][obj][index]['tem_rel'] = a['label']
                                label_rel_box[i][obj][index]['rel'].append([pred_labels_conv[j], 51, a['label']])
                                all_rels.append([pred_labels_conv[j], 51, a['label']])
        rel_obj_0 = np.array([[all_rels[k][0], str(k)+'_'+str(all_rels[k][1])] for k in range(len(all_rels))])
        rel_obj_1 = np.array([[str(k)+'_'+str(all_rels[k][1]), all_rels[k][2]] for k in range(len(all_rels))])
        pairs = np.concatenate((rel_obj_0, rel_obj_1))
        classes = list(set(pairs.flatten()))
        rel_num.append(len(classes))

        idx_map = {j: i for i, j in enumerate(classes)}
        edges = np.array(list(map(idx_map.get, pairs.flatten())), dtype=np.int32).reshape(pairs.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(idx_map), len(idx_map)), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = torch.zeros(adj.shape[0], len(self.idx2label)+len(self.idx2pred)+1)
        obj_set = {}
        for rel in pairs:
            for node in rel:
                if '-' in node:
                    obj = int(node.split('-')[1])
                    features[idx_map[node], obj-1] = 1
                else:
                    # frame-adj edge
                    rel = int(node.split('_')[1])
                    features[idx_map[node], len(self.idx2label)+rel-1] = 1


        features = sp.csr_matrix(features, dtype=np.float32)

        features = normalize_features(features)
        adj = normalize_adj(adj + sp.eye(adj.shape[0])) 


        return video_id, features, adj, rel_num

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

    # import pdb; pdb.set_trace()
    video_id, feature_bank, adj_bank, rel_num = zip(*data)

    return video_id, feature_bank, adj_bank, rel_num

def get_loader(batch_size, shuffle, num_workers, drop_last=False):
    dataset = SceneGraphDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                            num_workers=num_workers, collate_fn=collate_fn)
    
    return dataset, data_loader