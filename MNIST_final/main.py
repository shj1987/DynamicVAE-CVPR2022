"""main.py"""

import argparse

import numpy as np
import torch

# from solver_vae import Solver
from solver_dynamic import Solver

# from solver_emv import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available() ## if have gpu or cpu

def main(args):
    ## random seeds
    seed = args.seed

    if use_cuda:
        print("**using GPU training model**")
        torch.cuda.set_device(args.gpu)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    np.random.seed(seed)
    net = Solver(args)

    if args.train:
        net.train()
    else:
        net.viz_traverse(args.limit)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--gpu', default=1, type=int, help='gpu id')
    parser.add_argument('--C_start', default=0, type=float, help='start value of C')
    parser.add_argument('--past_T', default=10, type=int, help='past_T')
    parser.add_argument('--ramp_steps', default=1000, type=int, help='ramp steps')
    parser.add_argument('--Ki', default=-0.005, type=float, help='kp')
    parser.add_argument('--Kp', default=0.01, type=float, help='Ki')
    
    parser.add_argument('--max_iter', default=1e5, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--limit', default=3, type=float, help='traverse limits')
    parser.add_argument('--KL_loss', default=20, type=float, help='KL_divergence')
    parser.add_argument('--step_val', default=0.15, type=float, help='step_val')
    parser.add_argument('--pid_fixed', default=False, type=str2bool, help='if fixed PID or dynamic')
    parser.add_argument('--is_PID', default=True, type=str2bool, help='if use pid or not')
    
    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=120, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=7, type=int, help='dataloader num_workers')
    
    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--gather_step', default=50000, type=int, help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=50000, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=50000, type=int, help='number of iterations after which a checkpoint is saved')
    
    parser.add_argument('--ckpt_dir', default='checkpoints-PI', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
    
    args = parser.parse_args()
    
    main(args)
    
    
