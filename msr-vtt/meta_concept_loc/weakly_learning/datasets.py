import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
import random
from PIL import Image

def get_imgs(img_path_list, transform=None):
    ret = []
    for img_path in img_path_list:
        img_path = os.path.join('../../data/msrvtt_frames/', img_path)

        image = Image.open(img_path)
        img = transform(image) # (3, 256, 256)

        ret.append(img)

    imgs = np.stack(ret)
    return imgs

class CaptionDataset(Dataset):

    def __init__(self, split, transform=None):

        self.split = split
        assert self.split in {'train', 'val', 'test'}

        key_frame_list = os.path.join('../../data/metadata', 'frame_'+split+'_id_dict.json') 
        self.key_frame = json.load(open(key_frame_list))
        self.key_frame_list = list(self.key_frame.keys())

        self.num_frames = 4
        
        label_h5 = os.path.join('../../data/metadata', 'msrvtt_'+split+'_sequencelabel.h5') 
        self.label_h5 = h5py.File(label_h5, 'r')

        self.vocab = [i for i in self.label_h5['vocab']]
        self.videos = [i for i in self.label_h5['videos']]
        self.ixtoword = {i: w for i, w in enumerate(self.vocab)}
        self.wordtoix = {w: i for i, w in enumerate(self.vocab)}

        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.label_h5['label_start_ix'][:]
        self.label_end_ix = self.label_h5['label_end_ix'][:]
        assert(self.label_start_ix.shape[0] == self.label_end_ix.shape[0])

        self.captions = self.label_h5['labels'][:]

        self.transform = transform

        self.seq_per_img = 20


    def __getitem__(self, index):
        frame_id_list = self.key_frame[self.key_frame_list[index]]
        if len(frame_id_list) < self.num_frames:
            frame_id_list = [frame_id_list[i] for i in sorted(np.random.randint(len(frame_id_list), size=self.num_frames))]
        else:
            frame_id_list = [frame_id_list[i] for i in sorted(random.sample(range(len(frame_id_list)), self.num_frames))]

        cls_id = frame_id_list[0].split('/')[0]

        ## MSR-VTT
        video_id = int(cls_id[5:])
        if self.split == 'val':
            idx = video_id - 6513
        elif self.split == 'test':
            idx = video_id - 7010
        else:
            idx = video_id

        imgs = get_imgs(frame_id_list, self.transform)

        # import pdb; pdb.set_trace()
        ix1 = self.label_start_ix[idx]
        ix2 = self.label_end_ix[idx]
        ncap = ix2-ix1
        sent_ix = random.randint(0, ncap-1)  #seq_len = 20
        caption = torch.LongTensor(self.captions[ix1+sent_ix])

        try:
            eos_pos = list(caption).index(0)
            caplen = torch.LongTensor([eos_pos+1])
        except:
            caption[29] = 0
            caplen = torch.LongTensor([30])


        if self.split is 'train':
            return imgs, caption, caplen
        else:
            seq = torch.LongTensor(self.seq_per_img, 30).zero_()
            seq_all = torch.from_numpy(np.array(self.captions[ix1:ix2]))
            if ncap < self.seq_per_img:
                seq[:ncap] = seq_all[:ncap]
                for q in range(ncap, self.seq_per_img):
                    ix = np.random.randint(ncap)
                    seq[q] = seq_all[ix]

                all_captions = seq
            else:
                all_captions = seq_all
            return imgs, caption, caplen, all_captions

    def __len__(self):
        return len(self.key_frame_list)