# Cross-Modal Graph With Meta Concepts for Video Captioning

## Official PyTorch implementation
**Cross-Modal Graph With Meta Concepts for Video Captioning** <br>
*IEEE Transactions on Image Processing (TIP)* <br>
Hao Wang, Guosheng Lin, Steven C. H. Hoi, and Chunyan Miao <br>
[Paper](https://arxiv.org/pdf/2108.06458.pdf)

## Requirements
* pytorch 1.2 or higher
* python 3.6 or higher

```
git clone --recurse-submodules https://github.com/hwang1996/Meta-Concepts-for-Video-Captioning
```

## Dataset
Please download MSR-VTT dataset from [here](https://github.com/mynlp/cst_captioning) to use our codes.

## Preparation
- Extract video key frames
```
cd preprocess/
python extract_key_frames.py
```
- Use the weakly learning approach to produce meta concepts
```
cd meta_concept_loc/weakly_learning
python train.py
python generate_synonym.py
python extract_mask.py
```
- Train the segmentation model for meta concept inference
```
cd meta_concept_loc/segmentation
python train_custom.py
python extract_fea.py
```
- Please refer to this [repo](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to extract scene graphs

## Video captioning training

### Cross-entropy training
```
cd captioning
bash run_train.sh
```

### Reinforcement learning
```
bash run_rl_train.sh
```

## Testing
```
bash run_test.sh
```

## Acknowledgement

Our code builds upon several previous works:

- [a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
- [pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation)
- [cst_captioning](https://github.com/mynlp/cst_captioning)

## Reference
If you find this repository useful, please cite:
```
@article{wang2022cross,
  title={Cross-modal graph with meta concepts for video captioning},
  author={Wang, Hao and Lin, Guosheng and Hoi, Steven CH and Miao, Chunyan},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={5150--5162},
  year={2022},
  publisher={IEEE}
}
```
