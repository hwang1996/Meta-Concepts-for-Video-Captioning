import sys
import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    ### added for scene graph usage
    parser.add_argument(
        '--scene_graph_path',
        type=str,
        default='../data/MSR-VTT/files/',
        help='path to the pickle file containing scene graph information')
    parser.add_argument(
        '--total_node',
        type=int,
        default=200,
        help='maximum node number of scene graph')


    parser.add_argument(
        '--train_label_h5',
        type=str,
        default='output/metadata/msrvtt_train_sequencelabel.h5',
        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument(
        '--val_label_h5',
        type=str,
        default='output/metadata/msrvtt_val_sequencelabel.h5',
        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument(
        '--test_label_h5',
        type=str,
        default='output/metadata/msrvtt_test_sequencelabel.h5',
        help='path to the h5file containing the preprocessed dataset')

    parser.add_argument(
        '--train_feat_h5',
        type=str,
        nargs='+',
        default=[
            "output/feature/msrvtt_train_irv2_mp1.h5",
            # "output/feature/msrvtt_train_resnet_mp1.h5",
            "output/feature/msrvtt_train_c3d_mp1.h5",
            "output/feature/msrvtt_train_mfcc_mp1.h5",
            "output/feature/msrvtt_train_category_mp1.h5",
        ],
        help='path to the h5 file containing extracted features')
    parser.add_argument(
        '--val_feat_h5',
        type=str,
        nargs='+',
        default=[
            "output/feature/msrvtt_val_irv2_mp1.h5",
            # "output/feature/msrvtt_val_resnet_mp1.h5",
            "output/feature/msrvtt_val_c3d_mp1.h5",
            "output/feature/msrvtt_val_mfcc_mp1.h5",
            "output/feature/msrvtt_val_category_mp1.h5",
        ],
        help='path to the h5 file containing extracted features')
    parser.add_argument(
        '--test_feat_h5',
        type=str,
        nargs='+',
        default=[
            "output/feature/msrvtt_test_irv2_mp1.h5",
            # "output/feature/msrvtt_test_resnet_mp1.h5",
            "output/feature/msrvtt_test_c3d_mp1.h5",
            "output/feature/msrvtt_test_mfcc_mp1.h5",
            "output/feature/msrvtt_test_category_mp1.h5",
        ],
        help='path to the h5 file containing extracted features')

    parser.add_argument(
        '--train_cocofmt_file',
        type=str,
        default='output/metadata/msrvtt_train_cocofmt.json',
        help='Gold captions in MSCOCO format to cal language metrics')
    parser.add_argument(
        '--val_cocofmt_file',
        type=str,
        default='output/metadata/msrvtt_val_cocofmt.json',
        help='Gold captions in MSCOCO format to cal language metrics')
    parser.add_argument(
        '--test_cocofmt_file',
        type=str,
        default='output/metadata/msrvtt_test_cocofmt.json',
        help='Gold captions in MSCOCO format to cal language metrics')

    parser.add_argument(
        '--train_node_lmdb',
        type=str,
        # default='../data/MSR-VTT/files/msrvtt_seg_node_train',
        default='../data/MSR-VTT/files/msrvtt_seg_node_train_',
        help='path to the lmdb file containing the preprocessed segmentation output')
    parser.add_argument(
        '--val_node_lmdb',
        type=str,
        # default='../data/MSR-VTT/files/msrvtt_seg_node_val',
        default='../data/MSR-VTT/files/msrvtt_seg_node_val_',
        help='path to the lmdb file containing the preprocessed segmentation output')
    parser.add_argument(
        '--test_node_lmdb',
        type=str,
        # default='../data/MSR-VTT/files/msrvtt_seg_node_test',
        default='../data/MSR-VTT/files/msrvtt_seg_node_test_',
        help='path to the lmdb file containing the preprocessed segmentation output')


    parser.add_argument(
        '--train_bcmrscores_pkl',
        type=str,
        default='output/metadata/msrvtt_train_evalscores.pkl',
        help='Pre-computed Cider-D metric for all captions')
    
    # Optimization: General
    parser.add_argument(
        '--max_patience',
        type=int,
        default=50,
        help='max number of epoch to run since the minima is detected -- early stopping')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Video batch size (there will be x seq_per_img sentences)')
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=32,
        help='what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
    parser.add_argument(
        '--train_seq_per_img',
        type=int,
        default=20,
        help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive.')
    parser.add_argument(
        '--test_seq_per_img',
        type=int,
        default=20,
        help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        # default=1e-4,   ## batch size = 64
        default=8e-5,   ## batch size = 32
        help='learning rate')
    parser.add_argument(
        '--lr_update', default=1, type=int,
        help='Number of epochs to update the learning rate.')
    parser.add_argument(
        '--lr_decay_rate', default=1, type=float,
        help='decay rate of the learning rate.')

    # Model settings
    parser.add_argument(
        '--rnn_type',
        type=str,
        default='lstm',
        choices=[
            'lstm',
            'gru',
            'rnn'],
        help='type of RNN')
    parser.add_argument(
        '--rnn_size',
        type=int,
        default=512,
        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument(
        '--num_lm_layer',
        type=int,
        default=1,
        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument(
        '--input_encoding_size',
        type=int,
        default=512,
        help='the encoding size of each frame in the video.')
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=100,
        help='max number of epochs to run for (-1 = run forever)')
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=0.25,
        help='clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
    parser.add_argument(
        '--drop_prob_lm',
        type=float,
        default=0.5,
        help='strength of dropout in the Language Model RNN')

    # Optimization: for the Language Model
    parser.add_argument(
        '--optim',
        type=str,
        default='adam',
        help='what update to use? sgd|sgdmom|adagrad|adam')
    parser.add_argument(
        '--optim_alpha',
        type=float,
        default=0.8,
        help='alpha for adagrad/rmsprop/momentum/adam')
    parser.add_argument(
        '--optim_beta',
        type=float,
        default=0.999,
        help='beta used for adam')
    parser.add_argument(
        '--optim_epsilon',
        type=float,
        default=1e-8,
        help='epsilon that goes into denominator for smoothing')

    # Evaluation/Checkpointing
    parser.add_argument(
        '--save_checkpoint_from',
        type=int,
        default=1,
        help='Start saving checkpoint from this epoch')
    parser.add_argument(
        '--save_checkpoint_every',
        type=int,
        default=1,
        help='how often to save a model checkpoint in epochs?')

    parser.add_argument(
        '--use_rl',
        type=int,
        default=0,
        help='Use RL training or not')
    parser.add_argument(
        '--use_rl_after',
        type=int,
        default=0,
        help='Start RL training after this epoch')
    parser.add_argument(
        '--train_cached_tokens',
        type=str,
        default='output/metadata/msrvtt_train_ciderdf.pkl',
        help='Path to idx document frequencies to cal Cider on training data')
    parser.add_argument(
        '--expand_feat',
        type=int,
        default=1,
        help='To expand features when sampling (to multiple captions)')

    parser.add_argument('--checkpoint_path', type=str, help='output model file')
    parser.add_argument('--model_file', type=str, help='output model file')
    parser.add_argument('--best_model_file', type=str, help='output model file')
    parser.add_argument('--result_file', type=str, help='output result file')
    parser.add_argument(
        '--start_from',
        type=str,
        default='No',
        help='Load state from this file to continue training')
    parser.add_argument(
        '--language_eval',
        type=int,
        default=1,
        help='Evaluate language evaluation')
    parser.add_argument(
        '--eval_metric',
        default='CIDEr',
        choices=[
            'Loss',
            'Bleu_4',
            'METEOR',
            'ROUGE_L',
            'CIDEr',
            'MSRVTT'],
        help='Evaluation metrics')
    parser.add_argument(
        '--test_language_eval',
        type=int,
        default=1,
        help='Evaluate language evaluation')

    parser.add_argument(
        '--print_log_interval',
        type=int,
        default=20,
        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument(
        '--loglevel',
        type=str,
        default='INFO',
        choices=[
            'DEBUG',
            'INFO',
            'WARNING',
            'ERROR',
            'CRITICAL'])

    # misc
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='random number generator seed to use')
    parser.add_argument(
        '--gpuid',
        type=int,
        default=7,
        help='which gpu to use. -1 = use CPU')
    parser.add_argument(
        '--num_chunks',
        type=int,
        default=1,
        help='1: no attention, > 1: attention with num_chunks')
    parser.add_argument(
        '--feat_num_layers',
        type=int,
        default=1,
        help='number of layers in the feat pool ')
    parser.add_argument(
        '--rnn_num_layers',
        type=int,
        default=2,
        help='number of layers in the lstm ')

    parser.add_argument(
        '--model_type',
        type=str,
        default='concat',
        choices=[
            'standard',
            'concat',
            'manet',
            ],
        help='Type of models')
    
    parser.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='Beam search size')

    parser.add_argument(
        '--use_ss',
        type=int,
        default=0,
        help='Use schedule sampling')
    parser.add_argument(
        '--use_ss_after',
        type=int,
        default=0,
        help='Use schedule sampling after this epoch')
    parser.add_argument(
        '--ss_max_prob',
        type=float,
        default=0.25,
        help='Use schedule sampling')
    parser.add_argument(
        '--ss_k',
        type=float,
        default=100,
        help='plot k/(k+exp(x/k)) from x=0 to 400, k=30')

    parser.add_argument(
        '--use_mixer',
        type=int,
        default=0,
        help='Use schedule sampling')
    parser.add_argument(
        '--mixer_from',
        type=int,
        default=-1,
        help='If -1, then an annealing scheme will be used, based on mixer_descrease_every.\
        Initially it will set to the max_seq_length (30), and will be gradually descreased to 1.\
        If this value is set to 1 from the begininig, then the MIXER approach is not applied')
    parser.add_argument(
        '--mixer_descrease_every',
        type=int,
        default=2,
        help='Epoch interval to descrease mixing value')
    parser.add_argument(
        '--use_cst',
        type=int,
        default=0,
        help='Use cst training')
    parser.add_argument(
        '--use_cst_after',
        type=int,
        default=0,
        help='Start cst training after this epoch')
    parser.add_argument(
        '--cst_increase_every',
        type=int,
        default=5,
        help='Epoch interval to increase cst baseline')
    parser.add_argument(
        '--scb_baseline',
        type=int,
        default=1,
        help='which Self-consensus baseline (SCB) to use? 1: GT SCB, 2: Model Sample SCB')
    parser.add_argument(
        '--scb_captions',
        type=int,
        default=20,
        help='-1: annealing, otherwise using this fixed number to be the number of captions to compute SCB')
    parser.add_argument(
        '--use_eos',
        type=int,
        default=0,
        help='If 1, keep <EOS> in captions of the reference set')
    parser.add_argument(
        '--output_logp',
        type=int,
        default=0,
        help='Output average log likehood of the test and GT captions. Used for robustness analysis at test time.')
    
    
    args = parser.parse_args()
    return args
