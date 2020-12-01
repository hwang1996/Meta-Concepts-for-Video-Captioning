import json
from tqdm import tqdm
import torch
import numpy as np
import torch
import torch.nn as nn
from data_loader_local import *
import torch.nn.functional as F
import scipy.sparse as sp
import pickle


if __name__ == "__main__":

    ### give scene graph adj matrix
    sg_adj_all = {}

    rel_num_ = []

    dataset, data_loader = get_loader(batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    total_step = len(data_loader)
    loader = iter(data_loader)
    for i in tqdm(range(total_step)):
        video_id, feature_bank, adj_bank, rel_num = loader.next()
        video_id = video_id[0]

        rel_num_.extend(rel_num[0])
        
        sg_adj_all[video_id] = {}
        sg_adj_all[video_id]['feat'] = feature_bank[0]
        sg_adj_all[video_id]['adj'] = adj_bank[0]
        # import pdb; pdb.set_trace()


    #### max rel_num: 170, min rel_num: 1, most frequent: 2, mean: 11.3, median: 8
    #### max node_num: 19, min node_num: 2, most frequent: 2, mean: 4.83, median: 4
    print('dump the pickle file...')
    pickle.dump(sg_adj_all, open('../data/graph/adj_pair_edgenode_35.pkl', 'wb'))
