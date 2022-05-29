"""solver.py"""

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif, str2bool
from model import BetaVAE_H, BetaVAE_B
from dataset import return_data
from loss import reconstruction_loss, kl_divergence
from elbo_decomposition import elbo_decomposition


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[], 
                    beta=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.z_dim = args.z_dim
        self.model = args.model
        self.dataset = args.dataset
        
        if self.dataset == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif self.dataset == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif self.dataset == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif self.dataset == 'smallnorb':
            self.nc = 1
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        if self.model == 'H':
            net = BetaVAE_H
        elif self.model == 'B':
            net = BetaVAE_B
        else:
            raise NotImplementedError('only support model H or B')

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)

        self.viz_name = args.viz_name

        # load checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        self.ckpt_name = args.ckpt_name
        self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.gather = DataGather()

        self.net_mode(train=False)
        
    def kl_decomposition(self):
        _, analytical_cond_kl, \
        dependence, information, dimwise_kl, \
        _, _ = elbo_decomposition(self.net, self.data_loader)

        print("KLD:{} TC:{} MI:{} dKL:{}".format(analytical_cond_kl,
                                                 dependence,
                                                 information,
                                                 dimwise_kl))

        out_file = os.path.join(self.ckpt_dir, "KL_decomposition.txt")

        with open(out_file,"a") as fout:
            fout.write("checkpoint: " + self.ckpt_name + "\n")
            fout.write("KL: " + str(analytical_cond_kl.cpu().numpy()) + "\n")
            fout.write("TC: " + str(dependence.cpu().numpy()) + "\n")
            fout.write("MI: " + str(information.cpu().numpy()) + "\n")
            fout.write("dimension-wise kl: " + str(dimwise_kl.cpu().numpy()) + "\n")

    def reconstruction(self):
        i = 0
        n = 20
        for x in self.data_loader:
            i += 1
            if i == n:
                break

            x = Variable(cuda(x, self.use_cuda))
            x_recon, _, _ = self.net(x)

            self.gather.insert(images=x.data)
            self.gather.insert(images=torch.sigmoid(x_recon).data)

        x = self.gather.data['images'][0][:n]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:n]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        if self.save_output:
            save_image(tensor=images, fp=os.path.join(self.output_dir, 'recon.jpg'), pad_value=1)

    def traverse(self, limit=3, inter=6/9, loc=-1):
        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        with torch.no_grad():
            random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda))

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]
            
            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2, 'fixed_heart':fixed_img_z3}
        if self.dataset == 'smallnorb':
            fixed_idx1 = 5 # four-legged animal
            fixed_idx2 = 6 # human
            fixed_idx3 = 22 # airplane
            fixed_idx4 = 3 # truck
            fixed_idx5 = 4 # car

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            with torch.no_grad():
                fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda)).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            with torch.no_grad():
                fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda)).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            with torch.no_grad():
                fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda)).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)
            with torch.no_grad():
                fixed_img4 = Variable(cuda(fixed_img4, self.use_cuda)).unsqueeze(0)
            fixed_img_z4 = encoder(fixed_img4)[:, :self.z_dim]

            fixed_img5 = self.data_loader.dataset.__getitem__(fixed_idx5)
            with torch.no_grad():
                fixed_img5 = Variable(cuda(fixed_img5, self.use_cuda)).unsqueeze(0)
            fixed_img_z5 = encoder(fixed_img5)[:, :self.z_dim]
            
            Z = {'fixed_animal':fixed_img_z1, 'fixed_human':fixed_img_z2,
                 'fixed_airplane':fixed_img_z3, 'fixed_truck':fixed_img_z4,
                 'fixed_car':fixed_img_z5}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]
            Z = {'fixed_img':fixed_img_z, 'random_z':random_z}
            
        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val  ## row is the z latent variable
                    sample = torch.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal'.format(key)

        if self.save_output:
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(self.output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(self.output_dir, key+'*.jpg'),
                         os.path.join(self.output_dir, key+'.gif'), delay=10)

    def generate_image(self):
        n_dsets = len(self.data_loader.dataset)
        pbar = tqdm(total=n_dsets)
        for i in range(n_dsets):
            pbar.update(1)
            img = self.data_loader.dataset.__getitem__(i)
            img = Variable(cuda(img, self.use_cuda), volatile=True).unsqueeze(0)
            save_image(tensor=img.cpu(),
                    fp=os.path.join("smallNORB_images", '{}.jpg'.format(i)),
                    pad_value=1)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.net.load_state_dict(checkpoint['model_states']['net'])
            print("=> loaded checkpoint '{}'".format(file_path))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')
    parser.add_argument('--gpu', default=3, type=int, help='gpu id')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--limit', default=3, type=float, help='traverse limits')
    
    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='smallnorb', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')
    
    parser.add_argument('--viz_name', default='small_norb_13_PI90', type=str, help='visdom env name')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='qualitative_result', type=str, help='output directory')

    
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='1500000', type=str, help='load previous checkpoint. insert checkpoint filename')
    
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)
    
    print("--- KL decomposition --")
    net.kl_decomposition()
    print("--- reconstruction ---")
    net.reconstruction()
    print("--- traverse ---")
    net.traverse()
