# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import lmdb
import json
import pickle

# dynamic graph from knn
def knn(x, y=None, k=10):
    """
    :param x: BxCxN
    :param y: BxCxM
    :param k: scalar
    :return: BxMxk
    """
    if y is None:
        y = x
    # logging.info('Size in KNN: {} - {}'.format(x.size(), y.size()))
    # import pdb; pdb.set_trace()   
    inner = -2 * torch.matmul(y.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return idx


# get graph feature
def get_graph_feature(x, prev_x=None, k=20, idx_knn=None, r=-1, style=0):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    # import pdb; pdb.set_trace()
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_knn


# basic block
class GCNeXt(nn.Module):
    def __init__(self, channel_in, channel_out, k=3, norm_layer=None, groups=32, width_group=4):
        super(GCNeXt, self).__init__()
        self.k = k
        self.groups = groups

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = width_group * groups
        self.aconvs = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=1),
            # nn.Conv1d(channel_in, width, kernel_size=1), nn.ReLU(True),
            # nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
            # nn.Conv1d(width, channel_out, kernel_size=1),
        ) # adapation graph

        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1), nn.ReLU(True),
        ) # semantic graph

        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x  # residual
        # tout = self.tconvs(x)  # conv on temporal graph
        # import pdb; pdb.set_trace()

        x_f, idx = get_graph_feature(x, k=self.k, style=1)  # (bs,ch,100) -> (bs, 2ch, 100, k)
        sout = self.sconvs(x_f)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)

        out = self.aconvs(identity + sout)  # fusion

        return self.relu(out)



class GTAD(nn.Module):
    def __init__(self, k=3):
        super(GTAD, self).__init__()
        self.feat_dim = 2048
        self.h_dim_1d = 256
        self.h_dim_2d = 128
        self.h_dim_3d = 512

        self.k = k

        # Backbone Part 1
        self.backbone1 = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.h_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=self.k, groups=32),
        )

        # Backbone Part 2
        self.backbone2 = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=self.k, groups=32),
        )


    def forward(self, fused_fea):
        # import pdb; pdb.set_trace()
        base_feature = self.backbone1(fused_fea).contiguous()  # (bs, 2048, len) -> (bs, 256, len)
        gcnext_feature = self.backbone2(base_feature)  #

        return gcnext_feature


if __name__ == '__main__':
    env = lmdb.open('../../data/MSR-VTT/files/msrvtt_seg_node_train')
    txn = env.begin()
    frame_list = json.load(open('../../data/MSR-VTT/files/frame_train_mask_id.json'))
    max_len = 80
    label_bacth = torch.zeros(2, max_len, 60)
    fea_bacth = torch.zeros(2, max_len, 2048)

    label_emb = nn.Linear(60, 2048, bias=False).cuda()
    label_ada = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
        ).cuda()
    fea_ada = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
        ).cuda()
    for i in range(2):
        video_id = frame_list[i].split('/')[0]
        label, fea = pickle.loads(txn.get(video_id.encode()))
        label = torch.tensor(label.todense())

        fea_bacth[i] = fea
        label_bacth[i] = label

    label_bacth = label_bacth.cuda()
    fea_bacth = fea_bacth.cuda()
    label_fea = label_emb(label_bacth)
    label_fea = label_ada(label_fea)
    fea_bacth = fea_ada(fea_bacth)
    fused_fea = fea_bacth + label_fea

    model = GTAD().cuda()
    gcnext_feature = model(fused_fea.transpose(1, 2))
    import pdb; pdb.set_trace()
