import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
import h5py
import os
import cv2
import pickle
import lmdb
from tqdm import tqdm
import multiprocessing
from scipy import sparse
import spacy
nlp = spacy.load('en')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASS = 60
THRESHOLD = 150


def caption_image_beam_search(encoder, decoder, frame_id_list, caption, caplen, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    transform_re = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),   ## (width, height) -> (height,width)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    
    ret = []
    for image_path in frame_id_list:
        img_path = os.path.join('../../data/msrvtt_frames/', image_path)
        image = Image.open(img_path)
        image = transform_re(image).to(device) # (3, 256, 256)

        ret.append(image)
    imgs = torch.stack(ret)

    imgs = imgs.reshape(-1, imgs.size(-3), imgs.size(-2), imgs.size(-1))

    batch_size = 1
    img_len = 4

    # Encode
    encoder_out = encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
    ih, iw, ndf = encoder_out.size(-3), encoder_out.size(-2), encoder_out.size(-1)
    encoder_out = encoder_out.reshape(batch_size, img_len, ih, iw, ndf)
    encoder_out_new = encoder_out.new(batch_size, ih*int(img_len/2), iw*int(img_len/2), ndf)
    for i in range(batch_size):
        for j in range(img_len):
            h = int(j/int(img_len/2))
            w = j%int(img_len/2)
            encoder_out_new[i, ih*h:ih*(h+1), iw*w:iw*(w+1), :] = encoder_out[i, j]

    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder_out_new, caption, caplen)
    alphas = alphas.reshape(batch_size, -1, ih*int(img_len/2), iw*int(img_len/2))

    return caps_sorted.cpu().squeeze().numpy(), alphas.cpu().detach().squeeze()

def cmp_convert(alphas):
    alphas = (alphas - np.min(alphas)) / (np.max(alphas) - np.min(alphas))
    hmap = 1 - alphas
    hmap = (hmap * 255).astype(np.uint8)

    return hmap


def visualize_att(maps, labels, words, alphas, synonyms_class):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """

    for t in range(len(words)):
        try:
            word_lemma = [t.lemma_ for t in nlp(words[t])][0]
            label = synonyms_class[word_lemma]
        except KeyError:
            continue

        current_alpha = alphas[t]
        alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        alpha_ = cmp_convert(alpha)
        maps.append(alpha_)
        labels.append(label)

    return maps, labels

def mask_gen(cls_id):
    frame_id_list, alphas = pickle.loads(txn_alpha.get(cls_id.encode()))

    video_id = int(cls_id[5:])
    if split == 'val':
        idx = video_id - 6513
    elif split == 'test':
        idx = video_id - 7010
    else:
        idx = video_id

    ix1 = label_start_ix[idx]
    ix2 = label_end_ix[idx]
    ncap = ix2 - ix1

    att_sze = int(alphas.shape[-1]/int(num_frames/2))
    all_maps = []
    all_hard_mask = []
    all_soft_mask = []

    # for i, image in enumerate(ret):
    for i in range(num_frames):
        h = int(i/int(num_frames/2))
        w = i%int(num_frames/2)

        maps = []
        labels = []

        for sent_ix in range(ncap):
            caption = torch.LongTensor(captions[ix1+sent_ix])
            try:
                eos_pos = list(caption).index(0)
                caplen = eos_pos-1
            except:
                caplen = len(caption)-1
            
            current_alpha = alphas[sent_ix, :caplen, att_sze*h:att_sze*(h+1), att_sze*w:att_sze*(w+1)]
            # Visualize caption and attention of best sequence
            words = [rev_word_map[ind].decode() for ind in list(caption[1:caplen+1].numpy())]
            maps, labels = visualize_att(maps, labels, words, current_alpha, synonyms_class)

        ####################### all hard mask ################################
        # activated_map = np.stack(maps).min(axis=0)
        # activated_pos = np.stack(maps).argmin(axis=0)
        # map_class = np.where(activated_map < THRESHOLD, activated_map, 255)
        # for idx in range(len(labels)):
        #     mask = (activated_pos == idx)    # select the position for the idx'th map
        #     mask_map = (map_class < THRESHOLD)    # select the positions having minimum values (highest activations) 
        #     mask_ = np.where(mask == True, mask_map == mask, False) # the intersections
        #     map_class = np.where(mask_==True, labels[idx], map_class)
        # all_maps.append(map_class)

        ####################### soft mask ################################
        #### use union between masks
        activated_map_all = np.stack(maps)
        label_list = list(set(labels))
        soft_mask = np.zeros((activated_map_all.shape[-2], activated_map_all.shape[-1], NUM_CLASS)).astype(np.uint8)
        for label in label_list:
            pos = [l==label for l in labels]
            activated_map_ = activated_map_all[pos]
            activated_map_union = activated_map_.min(axis=0)

            map_class_soft = (255 - np.where(activated_map_union < THRESHOLD, activated_map_union, 255))
            soft_mask[:, :, label-1] = map_class_soft
        all_soft_mask.append(sparse.csr_matrix(soft_mask.reshape(-1, NUM_CLASS)))

    return frame_id_list, all_soft_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', default='BEST_checkpoint_msrvtt.pth.tar', help='path to model')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    synonyms_freq_list = pickle.load(open('../../data/metadata/synonyms_freq_list_60.pkl', 'rb'))
    synonyms_class = synonyms_freq_list['synonyms_class']

    # splits = ['train', 'val']
    splits = ['val']
    for split in splits:
        env = lmdb.open("../../data/graph/msrvtt_soft_mask_"+split, map_size=1099511627776)
        txn = env.begin(write = True)
        env_alpha = lmdb.open("../../data/graph/msrvtt_"+split+"_alpha")
        txn_alpha = env_alpha.begin(write = False)

        frame_id_all = []

        key_frame_list = os.path.join('../../data/metadata', 'frame_'+split+'_id_dict.json') 
        key_frame = json.load(open(key_frame_list))
        key_frame_list = list(key_frame.keys())
        num_frames = 4

        # Load word map (word2ix)
        label_h5 = os.path.join('../../data/metadata', 'msrvtt_'+split+'_sequencelabel.h5') 
        label_h5 = h5py.File(label_h5, 'r')
        vocab = [i for i in label_h5['vocab']]
        rev_word_map = {i: w for i, w in enumerate(vocab)}
        word_map = {w: i for i, w in enumerate(vocab)}

        label_start_ix = label_h5['label_start_ix'][:]
        label_end_ix = label_h5['label_end_ix'][:]
        assert(label_start_ix.shape[0] == label_end_ix.shape[0])
        captions = label_h5['labels'][:]

        with multiprocessing.Pool(processes=20) as p:
            max_ = len(key_frame_list)
            with tqdm(total=max_) as pbar:
                for _, (frame_id_list, all_soft_mask) in enumerate(p.imap_unordered(mask_gen, key_frame_list)):
                    frame_id_all.extend(frame_id_list)
                    for j in range(len(frame_id_list)):
                        txn.put(frame_id_list[j].encode(), pickle.dumps(all_soft_mask[j]))
                    pbar.update()
        # import pdb; pdb.set_trace()

        txn.commit()
        env.close()
        json.dump(frame_id_all, open('../../data/graph/frame_'+split+'_mask_id.json', 'w'))
