"""main.py"""

import argparse
import numpy as np
import torch
import os

from solver import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)


def main(args):
    print("\npid:", os.getpid())

    torch.cuda.set_device(args.gpu)
    net = Solver(args)

    if args.train:
        if args.model == 'vanilla':
            net.train_vanilla()
        elif args.model == 'beta':
            net.train_beta()
        elif args.model == 'factor':
            net.train_factor()
        elif args.model == 'ff':
            net.train_ff()
        elif args.model == 'factorS':
            net.train_factorS()
        elif args.model == 'sens_cls':
            net.train_sens_cls()
        elif args.model == 'ot_bl_cls':
            net.train_ot_bl_cls()
        elif args.model == 'ot_cls':
            net.train_ot_cls()

    if args.test:
        if args.model == 'vanilla' or args.model == 'factor' or \
            args.model == 'beta' or args.model == 'ff' or \
                args.model == 'factorS':
            net.test_vae()
        elif args.model == 'sens_cls':
            net.test_sens_cls()
        elif args.model == 'ot_bl_cls':
            net.test_ot_bl_cls()
        elif args.model == 'ot_cls':
            net.test_ot_cls()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE')

    parser.add_argument('--name', default='main', type=str,
                        help='name of the experiment')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='enable cuda')
    parser.add_argument('--max_iter', default=1e6, type=float,
                        help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')

    parser.add_argument('--z_dim', default=10, type=int,
                        help='dimension of the representation z')
    parser.add_argument('--gamma', default=6.4, type=float,
                        help='gamma hyperparameter')
    parser.add_argument('--beta', default=4.0, type=float,
                        help='beta hyperparameter')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='alpha hyperparameter')
    parser.add_argument('--phi', default=1.0, type=float,
                        help='phi hyperparameter')
    parser.add_argument('--eta', default=1.0, type=float,
                        help='eta hyperparameter')

    parser.add_argument('--lr_VAE', default=1e-4, type=float,
                        help='learning rate of the VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float,
                        help=('beta1 parameter of the Adam '
                              'optimizer for the VAE'))
    parser.add_argument('--beta2_VAE', default=0.999, type=float,
                        help=('beta2 parameter of the Adam '
                              'optimizer for the VAE'))
    parser.add_argument('--lr_D', default=1e-4, type=float,
                        help='learning rate of the discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float,
                        help=('beta1 parameter of the Adam '
                              'optimizer for the discriminator'))
    parser.add_argument('--beta2_D', default=0.9, type=float,
                        help=('beta2 parameter of the Adam '
                              'optimizer for the discriminator'))

    parser.add_argument('--dset_dir', default='data', type=str,
                        help='dataset directory')
    parser.add_argument('--dataset', default='CelebA', type=str,
                        help='dataset name')
    parser.add_argument('--image_size', default=64, type=int,
                        help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='dataloader num_workers')

    parser.add_argument('--print_iter', default=500, type=int,
                        help='print losses iter')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str,
                        help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None, type=str,
                        help='checkpoint name to load')
    parser.add_argument('--ckpt_save_iter', default=10000, type=int,
                        help='checkpoint save iter')

    parser.add_argument('--output_dir', default='outputs', type=str,
                        help='output directory')
    parser.add_argument('--output_save', default=True, type=str2bool,
                        help='whether to save traverse results')

    parser.add_argument('--gpu', default=0, type=int,
                        help='which gpu to use')
    parser.add_argument('--train', default=True, type=str2bool,
                        help='whether to train')
    parser.add_argument('--test', default=True, type=str2bool,
                        help='whether to test')
    parser.add_argument('--model', default='factor', type=str,
                        help='which vae')

    parser.add_argument('--n_sens', default=1, type=int,
                        help='the number of sensitive attributes')
    parser.add_argument('--sens_idx', nargs="+", type=int, default=[20],
                        help='indices of sensitive attributes')

    parser.add_argument('--ot_idx', default=18, type=int,
                        help='the index of original task label')

    parser.add_argument('--sens_cls_name', default=None, type=str,
                        help='sens_cls name')

    parser.add_argument('--sens_ckpt', default=None, type=str,
                        help='checkpoint name of sens_cls')

    args = parser.parse_args()

    main(args)
