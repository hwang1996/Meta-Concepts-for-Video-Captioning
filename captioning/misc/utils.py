import sys
import os
import json
from torch.nn.modules.loss import _WeightedLoss
import torch.nn as nn

import numpy as np
from collections import OrderedDict

sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append('coco-caption')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor

import pickle

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every [lr_update] epochs"""
    # lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    lr = opt.learning_rate * (opt.lr_decay_rate ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if isinstance(score, list):
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def load_gt_refs(cocofmt_file):
    d = json.load(open(cocofmt_file))
    out = {}
    for i in d['annotations']:
        out.setdefault(i['image_id'], []).append(i['caption'])
    return out


def compute_score(gt_refs, predictions, scorer):
    # use with standard package https://github.com/tylin/coco-caption
    # hypo = {p['image_id']: [p['caption']] for p in predictions}

    # use with Cider provided by https://github.com/ruotianluo/cider
    hypo = [{'image_id': p['image_id'], 'caption':[p['caption']]}
            for p in predictions]

    # standard package requires ref and hypo have same keys, i.e., ref.keys()
    # == hypo.keys()
    ref = {p['image_id']: gt_refs[p['image_id']] for p in predictions}

    score, scores = scorer.compute_score(ref, hypo)

    return score, scores


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[ix.item()].decode()
            else:
                break
        out.append(txt)
    return out

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def compute_avglogp(seq, logseq, eos_token=0):
    seq = seq.cpu().numpy()
    logseq = logseq.cpu().numpy()
    
    N, D = seq.shape
    out_avglogp = []
    for i in range(N):
        avglogp = []
        for j in range(D):
            ix = seq[i, j]
            avglogp.append(logseq[i, j])
            if ix == eos_token:
                break
        avg = 0 if len(avglogp) == 0 else sum(avglogp)/float(len(avglogp))
        out_avglogp.append(avg)
    return out_avglogp

def language_eval(gold_file, pred_file):

    # save the current stdout
    temp = sys.stdout
    # sys.stdout = open(os.devnull, 'w')

    coco = COCO(gold_file)
    cocoRes = coco.loadRes(pred_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = round(score, 5)

    # restore the previous stdout
    sys.stdout = temp
    return out


def array_to_str(arr, use_eos=0):
    out = ''
    for i in range(len(arr)):
        if use_eos == 0 and arr[i] == 0:
            break
        
        # skip the <bos> token    
        if arr[i] == 1:
            continue
            
        out += str(arr[i]) + ' '
        
        # return if encouters the <eos> token
        # this will also guarantees that the first <eos> will be rewarded
        if arr[i] == 0:
            break
            
    return out.strip()


def get_self_critical_reward2(model_res, greedy_res, gt_refs, scorer):

    model_score, model_scores = compute_score(model_res, gt_refs, scorer)
    greedy_score, greedy_scores = compute_score(greedy_res, gt_refs, scorer)
    scores = model_scores - greedy_scores

    m_score = np.mean(model_scores)
    g_score = np.mean(greedy_scores)

    #rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)

    return m_score, g_score


def get_self_critical_reward(
        model_res,
        greedy_res,
        data_gts,
        bcmr_scorer,
        expand_feat=0,
        seq_per_img=20,
        use_eos=0):
    
    batch_size = model_res.size(0)

    model_res = model_res.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    
    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [array_to_str(model_res[i], use_eos)]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i], use_eos)]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j], use_eos)
                  for j in range(len(data_gts[i]))]
    
    #_, scores = Bleu(4).compute_score(gts, res)
    #scores = np.array(scores[3])
    if isinstance(bcmr_scorer, CiderD):    
        res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
        
    if expand_feat == 1:
        gts = {i: gts[(i % batch_size) // seq_per_img]
               for i in range(2 * batch_size)}
    else:
        gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}

    score, scores = bcmr_scorer.compute_score(gts, res)
    
    # if bleu, only use bleu_4
    if isinstance(bcmr_scorer, Bleu):
        score = score[-1]
        scores = scores[-1]
    
    # happens for BLeu and METEOR
    if type(scores) == list:
        scores = np.array(scores)
    
    m_score = np.mean(scores[:batch_size])
    g_score = np.mean(scores[batch_size:])

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)

    return rewards, m_score, g_score


def get_cst_reward(
        model_res,
        data_gts,
        bcmr_scorer,
        bcmrscores=None,
        expand_feat=0,
        seq_per_img=20,
        scb_captions=20,
        scb_baseline=1,
        use_eos=0,
        use_mixer=0):
    
    """
    Arguments:
        bcmrscores: precomputed scores of GT sequences
        scb_baseline: 1 - use GT to compute baseline, 
                      2 - use MS to compute baseline
    """
    
    if bcmrscores is None or use_mixer == 1:
        batch_size = model_res.size(0)

        model_res = model_res.cpu().numpy()
        
        res = OrderedDict()
        for i in range(batch_size):
            res[i] = [array_to_str(model_res[i], use_eos)]

        gts = OrderedDict()
        for i in range(len(data_gts)):
            gts[i] = [array_to_str(data_gts[i][j], use_eos)
                      for j in range(len(data_gts[i]))]

        if isinstance(bcmr_scorer, CiderD):    
            res = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
        
        if expand_feat == 1:
            gts = {i: gts[(i % batch_size) // seq_per_img]
                   for i in range(batch_size)}
        else:
            gts = {i: gts[i % batch_size] for i in range(batch_size)}
        
        _, scores = bcmr_scorer.compute_score(gts, res)
            
        # if bleu, only use bleu_4
        if isinstance(bcmr_scorer, Bleu):
            scores = scores[-1]
    
        # happens for BLeu and METEOR
        if type(scores) == list:
            scores = np.array(scores)

        scores = scores.reshape(-1, seq_per_img)
            
    elif bcmrscores is not None and use_mixer == 0:
        # use pre-computed scores only when mixer is not used
        scores = bcmrscores.copy()
    else:
        raise ValueError('bcmrscores is not set!')
        
    if scb_captions > 0:
        
        sorted_scores = np.sort(scores, axis=1)
        
        if scb_baseline == 1:
            # compute baseline from GT scores
            sorted_bcmrscores = np.sort(bcmrscores, axis=1)
            m_score = np.mean(scores)
            b_score = np.mean(bcmrscores)
        elif scb_baseline == 2:
            # compute baseline from sampled scores
            m_score = np.mean(sorted_scores)
            b_score = np.mean(sorted_scores[:,:scb_captions])
        else:
            raise ValueError('unknown scb_baseline!')
        
        for ii in range(scores.shape[0]):
            if scb_baseline == 1:
                b = np.mean(sorted_bcmrscores[ii,:scb_captions])
            elif scb_baseline == 2:
                b = np.mean(sorted_scores[ii,:scb_captions])
            else:
                b = 0
            scores[ii] = scores[ii] - b
                
    else:
        m_score = np.mean(scores)
        b_score = 0
    
    scores = scores.reshape(-1)
    rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)
    
    return rewards, m_score, b_score


from collections import defaultdict, OrderedDict
import logging
import os
import re
import torch
import traceback

from torch.serialization import default_restore_location


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict


def save_state(filename, args, model, criterion, optimizer, lr_scheduler,
               num_updates, optim_history=None, extra_state=None):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    state_dict = {
        'args': args,
        'model': convert_state_dict_type(model.state_dict()),
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'optimizer_name': optimizer.__class__.__name__,
                'lr_scheduler_state': lr_scheduler.state_dict(),
                'num_updates': num_updates,
            }
        ],
        'last_optimizer_state': convert_state_dict_type(optimizer.state_dict()),
        'extra_state': extra_state,
    }
    torch_persistent_save(state_dict, filename)


def load_model_state(filename, model):
    if not os.path.exists(filename):
        return None, [], None
    state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    state = _upgrade_state_dict(state)
    model.upgrade_state_dict(state['model'])

    # load model parameters
    try:
        model.load_state_dict(state['model'], strict=True)
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')

    return state['extra_state'], state['optimizer_history'], state['last_optimizer_state']


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
    # add optimizer_history
    if 'optimizer_history' not in state:
        state['optimizer_history'] = [
            {
                'criterion_name': 'CrossEntropyCriterion',
                'best_loss': state['best_loss'],
            },
        ]
        state['last_optimizer_state'] = state['optimizer']
        del state['optimizer']
        del state['best_loss']
    # move extra_state into sub-dictionary
    if 'epoch' in state and 'extra_state' not in state:
        state['extra_state'] = {
            'epoch': state['epoch'],
            'batch_offset': state['batch_offset'],
            'val_loss': state['val_loss'],
        }
        del state['epoch']
        del state['batch_offset']
        del state['val_loss']
    # reduce optimizer history's memory usage (only keep the last state)
    if 'optimizer' in state['optimizer_history'][-1]:
        state['last_optimizer_state'] = state['optimizer_history'][-1]['optimizer']
        for optim_hist in state['optimizer_history']:
            del optim_hist['optimizer']
    # record the optimizer class name
    if 'optimizer_name' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['optimizer_name'] = 'FairseqNAG'
    # move best_loss into lr_scheduler_state
    if 'lr_scheduler_state' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['lr_scheduler_state'] = {
            'best': state['optimizer_history'][-1]['best_loss'],
        }
        del state['optimizer_history'][-1]['best_loss']
    # keep track of number of updates
    if 'num_updates' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['num_updates'] = 0
    # old model checkpoints may not have separate source/target positions
    if hasattr(state['args'], 'max_positions') and not hasattr(state['args'], 'max_source_positions'):
        state['args'].max_source_positions = state['args'].max_positions
        state['args'].max_target_positions = state['args'].max_positions
    # use stateful training data iterator
    if 'train_iterator' not in state['extra_state']:
        state['extra_state']['train_iterator'] = {
            'epoch': state['extra_state']['epoch'],
            'iterations_in_epoch': 0,
        }
    return state


def load_ensemble_for_inference(filenames, task, model_arg_overrides=None):
    """Load an ensemble of models for inference.
    model_arg_overrides allows you to pass a dictionary model_arg_overrides --
    {'arg_name': arg} -- to override model args that were used during model
    training
    """
    # load model architectures and weights
    states = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError('Model file not found: {}'.format(filename))
        state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        state = _upgrade_state_dict(state)
        states.append(state)
    args = states[0]['args']
    if model_arg_overrides is not None:
        args = _override_model_args(args, model_arg_overrides)

    # build ensemble
    ensemble = []
    for state in states:
        model = task.build_model(args)
        model.upgrade_state_dict(state['model'])
        model.load_state_dict(state['model'], strict=True)
        ensemble.append(model)
    return ensemble, args


def _override_model_args(args, model_arg_overrides):
    # Uses model_arg_overrides {'arg_name': arg} to override model args
    for arg_name, arg_val in model_arg_overrides.items():
        setattr(args, arg_name, arg_val)
    return args


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str):
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, 'r') as f:
            for line in f:
                cols = line.split()
                align_dict[cols[0]] = cols[1]
    else:
        # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
        # original source word.
        align_dict = {}
    return align_dict


def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    print("| Found {}/{} types in embedding file.".format(overlap, len(vocab_dict)))


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.
    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.
    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor([float(weight) for weight in pieces[1:]])
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding


def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
    from fairseq import tokenizer
    # Tokens are strings here
    hypo_tokens = tokenizer.tokenize_line(hypo_str)
    # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
    src_tokens = tokenizer.tokenize_line(src_str) + ['<eos>']
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return ' '.join(hypo_tokens)


def post_process_prediction(hypo_tokens, src_str, alignment, align_dict, tgt_dict, remove_bpe):
    from fairseq import tokenizer
    hypo_str = tgt_dict.string(hypo_tokens, remove_bpe)
    if align_dict is not None:
        hypo_str = replace_unk(hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string())
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tokenizer.Tokenizer.tokenize(hypo_str, tgt_dict, add_if_not_exist=True)
    return hypo_tokens, hypo_str, alignment


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def convert_padding_direction(src_tokens, padding_idx, right_to_left=False, left_to_right=False):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


def clip_grad_norm_(tensor, max_norm):
    grad_norm = item(torch.norm(tensor))
    if grad_norm > max_norm > 0:
        clip_coef = max_norm / (grad_norm + 1e-6)
        tensor.mul_(clip_coef)
    return grad_norm


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def checkpoint_paths(path, pattern=r'checkpoint(\d+)\.pt'):
    """Retrieves all checkpoints found in `path` directory.
    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = int(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]
