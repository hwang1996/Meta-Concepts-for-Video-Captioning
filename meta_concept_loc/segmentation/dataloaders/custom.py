from base import BaseDataSetCustom, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import lmdb
import json
import pickle

ignore_label = 255
ID_TO_TRAINID = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

class CustomDataset(BaseDataSetCustom):
    def __init__(self, mode='fine', **kwargs):
        self.num_classes = 60
        self.mode = mode
        self.palette = palette.CityScpates_palette
        self.id_to_trainId = ID_TO_TRAINID
        super(CustomDataset, self).__init__(**kwargs)

    def _set_files(self):

        self.images_base = os.path.join(self.root, 'msrvtt_frames')
        self.annotations_base = os.path.join(self.root, "graph/msrvtt_soft_mask_"+self.split)

        env = lmdb.open(self.annotations_base)
        self.txn = env.begin()

        self.files = json.load(open(os.path.join(self.root, 'graph/frame_'+self.split+'_mask_id.json')))

    def _load_data(self, index):

        img_path = os.path.join(self.images_base, self.files[index])
        image_id = self.files[index]
        image = Image.open(img_path).convert('RGB').resize([14 * 24, 14 * 24], Image.LANCZOS)
        image = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.float32)
        soft_mask = pickle.loads(self.txn.get(self.files[index].encode()))
        # soft_mask = pickle.loads(self.txn.get('video6688/000006.jpg'.encode()))
        # soft_mask = pickle.loads(self.txn.get('vid1252/000006.jpg'.encode()))
        soft_mask = np.asarray((soft_mask.toarray().reshape(14 * 24, 14 * 24, -1)), dtype=np.int32)

        label = [soft_mask[:, :, i] for i in range(self.num_classes)]
        return image, label, image_id



class Custom(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, mode='fine', val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = CustomDataset(mode=mode, **kwargs)
        super(Custom, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)


