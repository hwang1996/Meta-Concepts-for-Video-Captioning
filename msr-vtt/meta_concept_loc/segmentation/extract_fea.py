import os
import json
import argparse
import torch
import time
import dataloaders
import models
import inspect
import math
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer_custom import Trainer
from base import BaseTrainerCustom, DataPrefetcherCustom
import numpy as np
from utils.metrics_custom import eval_metrics, AverageMeter
from torchvision import transforms
from utils import transforms as local_transforms
from tqdm import tqdm
import lmdb
import torch.nn.functional as F
import pickle
from scipy import sparse

THRESHOLD = 0.3

env = lmdb.open("../video_text/data/MSR-VTT/files/msrvtt_node_val_", map_size=1099511627776)
txn = env.begin(write = True)

class Trainer(BaseTrainerCustom):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcherCustom(train_loader, device=self.device)
            self.val_loader = DataPrefetcherCustom(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True
        
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.))


    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        # tbar = tqdm(self.train_loader, ncols=130)
        tbar = tqdm(self.val_loader, ncols=130)
        count = 0
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target, _id) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                # import pdb; pdb.set_trace()
                # LOSS
                output, output_fea = self.model(data)

                # pos = torch.sigmoid(output) >= THRESHOLD
                output = torch.sigmoid(output)
                neg = (output <  THRESHOLD)
                # output[pos] = 1
                output[neg] = 0
                # output = F.interpolate(output, output_fea.shape[2:])
                for i in range(len(output)):
                    label = [j for j in range(len(output[i])) if output[i, j].sum() != 0]
                    gt_label = [j for j in range(len(target[i])) if 1 in target[i, j]]
                    # import pdb; pdb.set_trace()
                    if len(label) != 0:
                        select_fea = output[i][label].unsqueeze(1)
                        select_fea = output_fea[i] * select_fea
                        select_fea = select_fea.mean(-1).mean(-1)
                        # select_fea = {k: sparse.csr_matrix(select_fea[v].reshape(select_fea[v].shape[0], -1)) \
                        # 				for k in label for v in range(len(select_fea))}
                        # select_fea = [sparse.csr_matrix(select_fea[v].reshape(select_fea[v].shape[0], -1)) for v in range(len(select_fea))]
                        txn.put(_id[i].encode(), pickle.dumps([label, select_fea]))
                    else:
                        count += 1
            txn.commit()
            env.close()
            print(count)

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    trainer._valid_epoch(1)

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    # parser.add_argument('-r', '--resume', default='saved/PSPNet/10-25_07-48/checkpoint-epoch90.pth', type=str,
    # parser.add_argument('-r', '--resume', default='saved/PSPNet_msvd/11-03_22-32/checkpoint-epoch64.pth', type=str,
    parser.add_argument('-r', '--resume', default='saved/PSPNet/11-02_20-42/checkpoint-epoch50.pth', type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()



    config = json.load(open(args.config))
    # if args.resume:
    #     config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume)