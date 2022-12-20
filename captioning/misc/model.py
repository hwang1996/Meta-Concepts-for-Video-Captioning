import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.parameter import Parameter
import math
from layer import *
from dynamic_graph import *

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq, logprobs, reward):

        logprobs = to_contiguous(logprobs).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        # add one to the right to count for the <eos> token
        mask = to_contiguous(torch.cat(
            [mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - logprobs * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)

        return output


class CrossEntropyCriterion(nn.Module):

    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, pred, target, mask):
        # truncate to the same size
        target = target[:, :pred.size(1)]
        mask = mask[:, :pred.size(1)]

        pred = to_contiguous(pred).view(-1, pred.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        output = -pred.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class FeatPool(nn.Module):

    def __init__(self, feat_dims, out_size, dropout):
        super(FeatPool, self).__init__()

        module_list = []
        for dim in feat_dims:
            module = nn.Sequential(
                nn.Linear(
                    dim,
                    out_size),
                nn.ReLU(),
                nn.Dropout(dropout))
            module_list += [module]
        self.feat_list = nn.ModuleList(module_list)


    def forward(self, feats):
        """
        feats is a list, each element is a tensor that have size (N x C x F)
        at the moment assuming that C == 1
        """
        out = torch.cat([m(feats[i].squeeze(1))
                         for i, m in enumerate(self.feat_list)], 1)

        return out


class FeatExpander(nn.Module):

    def __init__(self, n=1):
        super(FeatExpander, self).__init__()
        self.n = n

    def forward(self, x):
        if self.n == 1:
            out = x
        else:
            out = Variable(
                x.data.new(
                    self.n * x.size(0),
                    x.size(1)))
            for i in range(x.size(0)):
                out[i * self.n:(i + 1) *
                    self.n] = x[i].expand(self.n, x.size(1))
        return out

    def set_n(self, x):
        self.n = x


class RNNUnit(nn.Module):

    def __init__(self, opt):
        super(RNNUnit, self).__init__()
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.rnn_num_layers = opt.rnn_num_layers
        self.drop_prob_lm = opt.drop_prob_lm

        if opt.model_type == 'standard':
            self.input_size = opt.input_encoding_size
        elif opt.model_type == 'concat':
            self.input_size = opt.input_encoding_size + self.rnn_size * 4

        self.rnn = getattr(
            nn,
            self.rnn_type.upper())(
            self.input_size,
            self.rnn_size,
            self.rnn_num_layers,
            bias=False,
            dropout=self.drop_prob_lm)

    def forward(self, xt, state):
        output, state = self.rnn(xt.unsqueeze(0), state)
        return output.squeeze(0), state



class DynamicGraph(nn.Module):
    def __init__(self, mc_cls, mc_size):
        super(DynamicGraph, self).__init__()

        self.seg_label_emb = nn.Linear(mc_cls, mc_size, bias=False)
        self.seg_label_ada = nn.Sequential(
                nn.Linear(mc_size, mc_size),
                nn.ReLU(),
            )
        self.seg_fea_ada = nn.Sequential(
                nn.Linear(mc_size, mc_size),
                nn.ReLU(),
            )

        self.model = GTAD()

    def forward(self, seg_fea, seg_label):
        seg_label_fea = self.seg_label_emb(seg_label)
        seg_label_fea = self.seg_label_ada(seg_label_fea)
        seg_vis_fea = self.seg_fea_ada(seg_fea)
        fused_seg_fea = seg_label_fea + seg_vis_fea

        gcnext_feature = self.model(fused_seg_fea.transpose(1, 2))
        return gcnext_feature.mean(-1)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, out_dim, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, out_dim, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x.mean(2)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerGraph(nn.Module):
    def __init__(self, opt, rnn_size):
        super(TransformerGraph, self).__init__()
        self.opt = opt
        self.rnn_size = int(rnn_size)

        self.graph_emb = GAT(nfeat=self.opt.total_node, 
                            nhid=8, 
                            out_dim=self.rnn_size, 
                            dropout=0.6, 
                            nheads=8, 
                            alpha=0.2)

        n_layers = 1
        n_heads = 4
        attn_drop = 0.2
        self.pos_encoder = PositionalEncoding(self.rnn_size, attn_drop)  ## self.rnn_size = 512
        encoder_layer = nn.TransformerEncoderLayer(self.rnn_size, n_heads, self.rnn_size, attn_drop)
        self.graph_emb_trans = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, sg_adj, sg_feat, sg_mask):

        sgs_feat = self.graph_emb(sg_feat, sg_adj)

        ### use transformer on the GCN features
        sgs_feat = sgs_feat.transpose(0, 1)
        sgs_feat = self.pos_encoder(sgs_feat)
        src_mask = self._generate_square_subsequent_mask(len(sgs_feat)).cuda()
        src_key_padding_mask = (sg_mask==0).bool()
        sgs_feat = self.graph_emb_trans(sgs_feat, src_mask, src_key_padding_mask=src_key_padding_mask) 
        sgs_feat = sgs_feat.transpose(0, 1)
        
        return sgs_feat

class CaptionModel(nn.Module):
    """
    A baseline captioning model
    """

    def __init__(self, opt):
        super(CaptionModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.rnn_num_layers = opt.rnn_num_layers
        self.feat_num_layers = opt.feat_num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)
        self.mc_size = opt.mc_size
        self.mc_cls = opt.mc_cls
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.bos_index = 1  # index of the <bos> token
        self.mixer_from = 0
        self.ss_prob = 0

        self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()
        self.feat_pool = FeatPool(
            self.feat_dims,
            self.feat_num_layers *
            self.rnn_size,
            self.drop_prob_lm)
        self.feat_expander = FeatExpander(self.seq_per_img)

        self.video_encoding_size = self.num_feats * self.feat_num_layers * self.rnn_size
        opt.video_encoding_size = self.video_encoding_size
        self.core = RNNUnit(opt)

        # ######################################################################################
        # #################### Add for scene graph integration #################################
        # ######################################################################################
        self.graph_fea = TransformerGraph(opt, self.rnn_size/2)

        self.sg_mapping = nn.Sequential(
            nn.Conv1d(6, 1, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(1, 1, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fc_mapping = nn.Sequential(
            nn.Linear((512*4), 2048),
            nn.ReLU(),
        )

        self.xt_mapping = nn.Sequential(
            nn.Linear(self.input_encoding_size, self.input_encoding_size),
            nn.ReLU(),
        )

        self.seg_graph = DynamicGraph(self.mc_cls, self.mc_size)
        self.seg_mapping = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.context_mapping = nn.Sequential(
            nn.Linear(256+256+2048, 2048),
            nn.ReLU(),
        )

        self.video_graph_emb = GAT(nfeat=opt.total_node + 1, 
                                nhid=4, 
                                out_dim=int(self.rnn_size/2), 
                                dropout=0.6, 
                                nheads=8, 
                                alpha=0.2)
        self.video_graph_mapping = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.graph_mapping = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
      
    def set_ss_prob(self, p):
        self.ss_prob = p

    def set_mixer_from(self, t):
        """Set values of mixer_from 
        if mixer_from > 0 then start MIXER training
        i.e:
        from t = 0 -> t = mixer_from -1: use XE training
        from t = mixer_from -> end: use RL training
        """
        self.mixer_from = t

    def set_seq_per_img(self, x):
        self.seq_per_img = x
        self.feat_expander.set_n(x)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if self.rnn_type == 'lstm':
            return (
                Variable(
                    weight.new(
                        self.rnn_num_layers,
                        batch_size,
                        self.rnn_size).zero_()),
                Variable(
                    weight.new(
                        self.rnn_num_layers,
                        batch_size,
                        self.rnn_size).zero_()))
        else:
            return Variable(
                weight.new(
                    self.rnn_num_layers,
                    batch_size,
                    self.rnn_size).zero_())

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, feats, seq, frame_sg, video_sg, seg_out):
        sg_adj, sg_feat, sg_mask = frame_sg
        sg_adj_g, sg_feat_g = video_sg
        seg_fea, seg_label = seg_out

        fc_feats = self.feat_pool(feats)
        fc_feats = self.fc_mapping(fc_feats)
        fc_feats = self.feat_expander(fc_feats)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        sample_seq = []
        sample_logprobs = []

        # -- if <image feature> is input at the first step, use index -1
        # -- the <eos> token is not used for training
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = seq.size(1) - 1

        
        bs = int(batch_size/self.seq_per_img)
        seg_out_fea = self.seg_graph(seg_fea, seg_label)
        seg_out_fea = self.seg_mapping(seg_out_fea)
        seg_fea = seg_out_fea.unsqueeze(1).expand(bs, self.seq_per_img, seg_out_fea.size(-1)).reshape(-1, seg_out_fea.size(-1))

        ### use transformer on the GCN features
        sgs_feat = self.graph_fea(sg_adj, sg_feat, sg_mask)
        sgs_feat = self.sg_mapping(sgs_feat).squeeze(1)
        video_sg_feat = self.video_graph_emb(sg_feat_g.unsqueeze(1), sg_adj_g.unsqueeze(1)).squeeze()
        video_sg_feat = self.video_graph_mapping(video_sg_feat)
        all_sg_feat = self.graph_mapping(torch.cat([sgs_feat, video_sg_feat], 1))
        all_sg_feat = all_sg_feat.unsqueeze(1).expand(bs, self.seq_per_img, all_sg_feat.size(-1)).reshape(-1, all_sg_feat.size(-1))

        for token_idx in range(start_i, end_i):
            if token_idx == -1:
                xt = fc_feats
            else:
                # token_idx = 0 corresponding to the <BOS> token
                # (already encoded in seq)

                if self.training and token_idx >= 1 and self.ss_prob > 0.0:
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, token_idx].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, token_idx].data.clone()
                        # fetch prev distribution: shape Nx(M+1)
                        prob_prev = torch.exp(outputs[-1].data)
                        sample_ind_tokens = torch.multinomial(
                            prob_prev, 1).view(-1).index_select(0, sample_ind)
                        it.index_copy_(0, sample_ind, sample_ind_tokens)
                        it = Variable(it, requires_grad=False)
                elif self.training and self.mixer_from > 0 and token_idx >= self.mixer_from:
                    prob_prev = torch.exp(outputs[-1].data)
                    it = torch.multinomial(prob_prev, 1).view(-1)
                    it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, token_idx].clone()
                
                # it = seq[:, token_idx].clone()

                if token_idx >= 1:
                    # store the seq and its logprobs
                    sample_seq.append(it.data)
                    logprobs = outputs[-1].gather(1, it.unsqueeze(1))
                    sample_logprobs.append(logprobs.view(-1))
                
                # break if all the sequences end, which requires EOS token = 0
                if it.data.sum() == 0:
                    break
                xt = self.embed(it)

            xt = self.xt_mapping(xt)
            # import pdb; pdb.set_trace()
            context = self.context_mapping(torch.cat([fc_feats, all_sg_feat, seg_fea], 1))
            output, state = self.core(torch.cat([xt, context], 1), state)
                
            if token_idx >= 0:
                output = F.log_softmax(self.logit(self.dropout(output)), dim=-1)
                outputs.append(output)
                
        # only returns outputs of seq input
        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), \
                torch.cat([_.unsqueeze(1) for _ in sample_seq], 1), \
                torch.cat([_.unsqueeze(1) for _ in sample_logprobs], 1) \


    def sample(self, feats, frame_sg, video_sg, seg_out, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        expand_feat = opt.get('expand_feat', 0)

        # import pdb; pdb.set_trace()

        if beam_size > 1:
            return self.sample_beam(feats, frame_sg, video_sg, seg_out, opt)

        fc_feats = self.feat_pool(feats)
        if expand_feat == 1:
            fc_feats = self.feat_expander(fc_feats)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        seq = []
        seqLogprobs = []

        unfinished = fc_feats.data.new(batch_size).fill_(1).byte()

        sg_adj, sg_feat, sg_mask = frame_sg
        sg_adj_g, sg_feat_g = video_sg
        seg_fea, seg_label = seg_out

        ### use transformer on the GCN features
        sgs_feat = self.graph_fea(sg_adj, sg_feat, sg_mask)
        if expand_feat == 1:
            sgs_feat = sgs_feat.repeat(self.seq_per_img, 1, 1)
            seg_fea = seg_fea.repeat(self.seq_per_img, 1)
        
        seg_fea = self.seg_linear(seg_fea)
        seg_fea = self.seg_mapping(seg_fea.unsqueeze(1)).squeeze(1)
        sgs_feat_p = self.sg_mapping(sgs_feat).squeeze(1)
        fc_feats = self.fc_mapping(fc_feats.unsqueeze(1)).squeeze(1)

        # -- if <image feature> is input at the first step, use index -1
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = self.seq_length - 1

        for token_idx in range(start_i, end_i):
            if token_idx == -1:
                xt = fc_feats
            else:
                if token_idx == 0:  # input <bos>
                    it = fc_feats.data.new(
                        batch_size).long().fill_(self.bos_index)
                elif sample_max == 1:
                    # output here is a Tensor, because we don't use backprop
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        # fetch prev distribution: shape Nx(M+1)
                        prob_prev = torch.exp(logprobs.data).cpu()
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(
                            torch.div(
                                logprobs.data,
                                temperature)).cpu()
                    #import pdb; pdb.set_trace()
                    it = torch.multinomial(prob_prev, 1).cuda()
                    # gather the logprobs at sampled positions
                    sampleLogprobs = logprobs.gather(
                        1, Variable(it, requires_grad=False))
                    # and flatten indices for downstream processing
                    it = it.view(-1).long()

                xt = self.embed(Variable(it, requires_grad=False))

            if token_idx >= 1:
                unfinished = unfinished * (it > 0).byte()

                #
                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

                # requires EOS token = 0
                if unfinished.sum() == 0:
                    break

            xt = self.xt_mapping(xt.unsqueeze(1)).squeeze(1)
            output, state = self.core(torch.cat([xt, fc_feats, sgs_feat_p, seg_fea], 1), state)

            logprobs = F.log_softmax(self.logit(output), dim=-1)

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat(
            [_.unsqueeze(1) for _ in seqLogprobs], 1)

    def sample_beam(self, feats, frame_sg, video_sg, seg_out, opt={}):
        """
        modified from https://github.com/ruotianluo/self-critical.pytorch
        """
        sg_adj, sg_feat, sg_mask = frame_sg
        sg_adj_g, sg_feat_g = video_sg
        seg_fea, seg_label = seg_out

        beam_size = opt.get('beam_size', 5)
        fc_feats = self.feat_pool(feats)
        batch_size = fc_feats.size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        fc_feats = self.fc_mapping(fc_feats)

        seg_out_fea = self.seg_graph(seg_fea, seg_label)
        seg_out_fea = self.seg_mapping(seg_out_fea)

        sgs_feat = self.graph_fea(sg_adj, sg_feat, sg_mask)
        sgs_feat = self.sg_mapping(sgs_feat).squeeze(1)
        video_sg_feat = self.video_graph_emb(sg_feat_g.unsqueeze(1), sg_adj_g.unsqueeze(1)).squeeze()
        video_sg_feat = self.video_graph_mapping(video_sg_feat)
        all_sg_feat = self.graph_mapping(torch.cat([sgs_feat, video_sg_feat], 1))

        # import pdb; pdb.set_trace()
            
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            fc_feats_k = fc_feats[k].expand(
                beam_size, 2048)
            all_sgs_feat_k = all_sg_feat[k].expand(beam_size, int(self.rnn_size/2))
            seg_fea_k = seg_out_fea[k].expand(beam_size, int(self.rnn_size/2))

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(
                self.seq_length, beam_size).zero_()
            # running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)

            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.seq_length - 1

            for token_idx in range(start_i, end_i):
                if token_idx == -1:
                    xt = fc_feats_k
                elif token_idx == 0:  # input <bos>
                    it = fc_feats.data.new(
                        beam_size).long().fill_(self.bos_index)
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    # sorted array of logprobs along each previous beam (last
                    # true = descending)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))  # ys.size(1)=10536 size of the vocabulary dict
                    rows = beam_size
                    if token_idx == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in
                            # (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append({'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.item(), 'r': local_logprob.item()})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # import pdb; pdb.set_trace()
                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if token_idx > 1:
                        # well need these as reference when we fork beams
                        # around
                        beam_seq_prev = beam_seq[:token_idx - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:token_idx - 1].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if token_idx > 1:
                            beam_seq[:token_idx - 1, vix] \
                                = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:token_idx - 1, vix] \
                                = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at
                            # vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        # c'th word is the continuation
                        beam_seq[token_idx - 1, vix] = v['c']
                        beam_seq_logprobs[token_idx - 1, vix] = v['r']  # the raw logprob here
                        # the new (sum) logprob along this beam
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 0 or token_idx == self.seq_length - 2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            if token_idx > 1: 
                                ppl = np.exp(-beam_logprobs_sum[vix] / (token_idx - 1))
                            else:
                                ppl = 10000
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix],
                                                       'ppl': ppl 
                                                       })

                    # encode as vectors
                    it = beam_seq[token_idx - 1]
                    xt = self.embed(Variable(it.cuda()))

                if token_idx >= 1:
                    state = new_state

  
                xt = self.xt_mapping(xt)
                context = self.context_mapping(torch.cat([fc_feats_k, all_sgs_feat_k, seg_fea_k], 1))
                output, state = self.core(
                    torch.cat([xt, context], 1), state)

                logprobs = F.log_softmax(self.logit(output), dim=-1)
                # logprobs = F.log_softmax(self.logit(output)/T, dim=-1)

            
            #self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            self.done_beams[k] = sorted(
                self.done_beams[k], key=lambda x: x['ppl'])
            
            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
            
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)
