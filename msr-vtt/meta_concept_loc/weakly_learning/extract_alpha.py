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
from dataloader import CaptionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', default='BEST_checkpoint_tri_msrvtt.pth.tar', help='path to model')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()
    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),   ## (width, height) -> (height,width)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = CaptionDataset('train', transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
    val_dataset = CaptionDataset('val', transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)

    env = lmdb.open("../../data/graph/msrvtt_train_alpha", map_size=1099511627776)
    txn = env.begin(write = True)

    for step, (frame_id_list, imgs, caps, caplens) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        batch_size = imgs.size(0)
        img_len = imgs.size(1)
        imgs = imgs.reshape(-1, imgs.size(-3), imgs.size(-2), imgs.size(-1))
        ncap = caps.size(1)
        caps = caps.reshape(-1, caps.size(-1))
        caplens = caplens.reshape(-1, caplens.size(-1))

        # Forward prop.
        imgs = encoder(imgs)
        ih, iw, ndf = imgs.size(-3), imgs.size(-2), imgs.size(-1)
        imgs = imgs.reshape(batch_size, img_len, ih, iw, ndf)
        imgs_new = imgs.new(batch_size, ih*int(img_len/2), iw*int(img_len/2), ndf)
        for i in range(batch_size):
            for j in range(img_len):
                h = int(j/int(img_len/2))
                w = j%int(img_len/2)
                imgs_new[i, ih*h:ih*(h+1), iw*w:iw*(w+1), :] = imgs[i, j]    # torch.Size([16, 28, 28, 2048])
        imgs_re = imgs_new.unsqueeze(1).repeat(1, 20, 1, 1, 1).reshape(-1, 28, 28, 2048)
        with torch.no_grad():
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_re, caps, caplens, phase='val')
        _, idx = torch.sort(sort_ind)
        alphas = alphas.reshape(batch_size*ncap, -1, ih*int(img_len/2), iw*int(img_len/2))
        alphas_o = alphas[idx].cpu().numpy()

        for frame_idx in range(len(frame_id_list[0])):
            video_id = frame_id_list[0][frame_idx].split('/')[0]
            frame_list = [frame_id_list[i][frame_idx] for i in range(img_len)]
            # import pdb; pdb.set_trace() 
            txn.put(video_id.encode(), \
                pickle.dumps([frame_list, alphas_o[frame_idx*ncap:(frame_idx+1)*ncap]]))
    txn.commit()
    env.close()

