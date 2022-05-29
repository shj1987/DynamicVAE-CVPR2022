"""solver.py"""

import os
import warnings
warnings.filterwarnings("ignore")

from collections import deque
import numpy as np
from tqdm import tqdm
import visdom
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif
from model import BetaVAE_H, BetaVAE_B
from dataset import return_data
from I_PID import PIDControl
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
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_max_org = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        # self.KL_loss = args.KL_loss
        self.pid_fixed = args.pid_fixed
        self.is_PID = args.is_PID
        self.step_value = args.step_val
        self.C_start = args.C_start
        self.ramp_steps = args.ramp_steps
        self.Kp = args.Kp
        self.Ki = args.Ki
        self.iv = args.iv
        self.period = args.past_T
        self.window_len = args.window_len
        self.dataset = args.dataset
        
        if args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'smallnorb':
            self.nc = 1
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        if args.model == 'H':
            net = BetaVAE_H
        elif args.model == 'B':
            net = BetaVAE_B
        else:
            raise NotImplementedError('only support model H or B')

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_beta = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.kld_recent = np.zeros((10, ))
        self.kld_stable = np.zeros((1000, ))

        self.gather = DataGather()
        self.gather2 = DataGather()

        self.tensorboard_on = args.tensorboard_on
        if self.tensorboard_on:
            self.log_dir = os.path.join(args.log_dir, args.viz_name)
            self.writer = SummaryWriter(self.log_dir)


    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False
        
        # pbar = tqdm(total=self.max_iter)
        # pbar.update(self.global_iter)
        ## write log to log file
        outfile = os.path.join(self.ckpt_dir, "train.log")
        kl_file = os.path.join(self.ckpt_dir, "train.kl")
        fw_log = open(outfile, "w")
        fw_kl = open(kl_file, "w")
        # fw_kl.write('total KL\tz_dim' + '\n')

        ## init PID control
        PID = PIDControl(self.Kp, self.Ki, self.iv)
        C = self.C_start
        window_len = self.window_len
        ramp_steps = self.ramp_steps
        queue = deque([0.] * (self.period-1))
        
        fw_log.write("Kp:{0:.5f} Ki:{1:.6f} C_iter:{2:.1f} period:{3} step_val:{4:.4f} batch:{5:.4f} ramp_steps:{6:.4f} past_T: {7:2f}\n" \
                    .format(self.Kp, self.Ki, self.C_stop_iter, window_len, self.step_value, self.batch_size, self.ramp_steps, self.period))
        fw_log.flush()

        pbar = tqdm(total=self.max_iter)

        config_file = os.path.join(self.ckpt_dir, "config.txt")

        with open(config_file, "a") as conf:
            if self.is_PID and self.objective == 'H':
                conf.write("main_model:DynamicVAE")
            elif self.objective == 'H':
                conf.write("main_model:BetaVAE-H")
            elif self.objective == 'B':
                conf.write("main_model:BetaVAE-B")
            conf.write("\n")

            conf.write("dataset:" + self.dataset + ' ')
            conf.write("batch_size:" + str(self.batch_size))
            conf.write("\n")

            conf.write("Optimizer:Adam" + ' ')
            conf.write("lr:" + str(self.lr) + ' ')
            conf.write("beta1:" + str(self.beta1) + ' ')
            conf.write("beta2:" + str(self.beta2))
            conf.write("\n")

            if self.is_PID and self.objective == 'H':
                conf.write("C_start:" + str(self.C_start) + ' ')
                conf.write("C_max:" + str(self.C_max.item()) + ' ')
                conf.write("step_value:" + str(self.step_value) + ' ')
                conf.write("remp_steps:" + str(self.ramp_steps) + ' ')
                conf.write("window_len:" + str(self.window_len) + ' ')
                conf.write("past_T:" + str(self.period))
            elif self.objective == 'H':
                conf.write("don't change C manually")
            elif self.objective == 'B':
                conf.write("C_start:" + str(self.C_start) + ' ')
                conf.write("C_max:" + str(self.C_max.item()) + ' ')
                conf.write("C_stop_iter:" + str(self.C_stop_iter))
                conf.write("main_model:BetaVAE-B")
            conf.write("\n")
            
            conf.write("z_dim:" + str(self.z_dim) + ' ')
            conf.write("max_iter:" + str(int(self.max_iter)))
            conf.write("\n")
            if self.is_PID and self.objective == 'H':
                conf.write("controller:I_PID" + ' ')
                conf.write("Kp:" + str(self.Kp) + ' ')
                conf.write("Ki:" + str(self.Ki) + ' ')
                conf.write("initial_value:" + str(self.iv))
            elif self.objective == 'H':
                conf.write("beta:" + str(self.beta))
            elif self.objective == 'B':
                conf.write("gamma:" + str(self.gamma))

        while not out:
            for x in self.data_loader:
                self.global_iter += 1

                pbar.update(1)

                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                self.kld_stable[self.global_iter % len(self.kld_stable)] = total_kld.item()


                if self.is_PID and self.objective == 'H': # utilize this one
                    if self.global_iter % window_len < ramp_steps and self.global_iter >= window_len:
                        C += self.step_value/ramp_steps
                    ## get the C max
                    if C > self.C_max_org:
                        C = self.C_max_org
                    queue.append(total_kld.item())
                    avg_kl = np.mean(queue)
                    self.beta, _ = PID.pid(C, avg_kl)
                    queue.popleft()
                
                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta * total_kld
                elif self.objective == 'B':
                    ### tricks for C
                    if self.is_PID:
                        if self.global_iter % window_len < ramp_steps and self.global_iter >= window_len:
                            C += self.step_value/ramp_steps
                        ## get the C max
                        if C > self.C_max_org:
                            C = self.C_max_org
                        C = torch.tensor(C).cuda()
                    else:
                        C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, self.C_start, self.C_max.data[0])
                    
                    beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()
                
                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data, beta=self.beta)
                
                if self.global_iter%20 == 0:
                    ## write log to file
                    if self.objective == 'B':
                        C = C.item()
                    fw_log.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} exp_kld:{:.3f} beta:{:.4f}\n'.format(
                                self.global_iter, recon_loss.item(), total_kld.item(), C, self.beta))
                    ## write KL to file
                    dim_kl_np = dim_wise_kld.data.cpu().numpy()
                    dim_kl_str = [str(k) for k in dim_kl_np]
                    fw_kl.write('total_kld:{0:.3f}\t'.format(total_kld.item()))
                    fw_kl.write('z_dim:' + ','.join(dim_kl_str) + '\n')
                    
                    if self.global_iter%500 == 0:
                        fw_log.flush()
                        fw_kl.flush()
                    
                if self.viz_on and self.global_iter % self.gather_step==0:
                    self.gather.insert(images=x.data)
                    self.gather.insert(images=F.sigmoid(x_recon).data)
                    self.viz_reconstruction()
                    self.viz_lines()
                    self.gather.flush()

                if (self.viz_on or self.save_output) and self.global_iter%200000==0:
                    self.viz_traverse()

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')
                    # pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                    
                if self.global_iter%200000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

            # log after a epoch
            if self.tensorboard_on:
                self.viz_log(C, recon_loss, total_kld, dim_kl_np)

        # pbar.write("[Training Finished]")
        # pbar.close()
        fw_log.close()
        fw_kl.close()
        

    def viz_log(self, C, recon_loss, total_kl, dim_kl):
        self.net_mode(train=False)

        self.writer.add_scalar('hyper/exp_kld', C, self.global_iter)
        self.writer.add_scalar('hyper/beta', self.beta, self.global_iter)
        self.writer.add_scalar('loss/recon_loss', recon_loss.item(), self.global_iter)
        self.writer.add_scalar('loss/total_kld', total_kl.item(), self.global_iter)
        for i in range(len(dim_kl)):
            self.writer.add_scalar('kld/{}'.format(i), dim_kl[i], self.global_iter)

        logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
            elbo_decomposition(self.net, self.data_loader)
        
        self.writer.add_scalar('crit/rec', -logpx, self.global_iter)
        self.writer.add_scalar('crit/kld', analytical_cond_kl, self.global_iter)
        self.writer.add_scalar('crit/TC', dependence, self.global_iter)
        self.writer.add_scalar('crit/MI', information, self.global_iter)
        self.writer.add_scalar('crit/dKL', dimwise_kl, self.global_iter)

        self.net_mode(train=True)


    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            save_image(tensor=images, fp=os.path.join(output_dir, 'recon.jpg'), pad_value=1)
        self.net_mode(train=True)


    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        betas = torch.Tensor(self.gather.data['beta'])

        # mus = torch.stack(self.gather.data['mu']).cpu()
        # vars = torch.stack(self.gather.data['var']).cpu()
        
        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        # mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        ## legend
        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))
        # legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        
        if self.win_beta is None:
            self.win_beta = self.viz.line(
                                        X=iters,
                                        Y=betas,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='beta',))
        else:
            self.win_beta = self.viz.line(
                                        X=iters,
                                        Y=betas,
                                        env=self.viz_name+'_lines',
                                        win=self.win_beta,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='beta',))

        if self.win_kld is None:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))
        else:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        win=self.win_kld,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))
        
        self.net_mode(train=True)


    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)
        
        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

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
            
            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        elif self.dataset.lower() == 'celeba':
            fixed_idx1 = 191281 # 'CelebA/img_align_celeba/191282.jpg'
            fixed_idx2 = 143307 # 'CelebA/img_align_celeba/143308.jpg'
            fixed_idx3 = 101535 # 'CelebA/img_align_celeba/101536.jpg'
            fixed_idx4 = 70059  # 'CelebA/img_align_celeba/070060.jpg'

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = encoder(fixed_img4)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'fixed_4':fixed_img_z4,
                 'random':random_img_z}
        elif self.dataset.lower() == '3dchairs':
            fixed_idx1 = 40919 # 3DChairs/images/4682_image_052_p030_t232_r096.png
            fixed_idx2 = 5172  # 3DChairs/images/14657_image_020_p020_t232_r096.png
            fixed_idx3 = 22330 # 3DChairs/images/30099_image_052_p030_t232_r096.png

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'random':random_img_z}
        elif self.dataset.lower() == 'smallnorb':
            fixed_idx1 = 5 # four-legged animal
            fixed_idx2 = 6 # human
            fixed_idx3 = 22 # airplane
            fixed_idx4 = 3 # truck
            fixed_idx5 = 4 # car

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = encoder(fixed_img4)[:, :self.z_dim]

            fixed_img5 = self.data_loader.dataset.__getitem__(fixed_idx5)
            fixed_img5 = fixed_img5.to(self.device).unsqueeze(0)
            fixed_img_z5 = encoder(fixed_img5)[:, :self.z_dim]
            
            Z = {'fixed_animal':fixed_img_z1, 'fixed_human':fixed_img_z2,
                 'fixed_airplane':fixed_img_z3, 'fixed_truck':fixed_img_z4,
                 'fixed_car':fixed_img_z5, 'random':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]
            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}
            
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
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
            
            if self.viz_on:
                self.viz.images(samples, env=self.viz_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            save_image(tensor=fixed_img1.cpu(),
                       fp=os.path.join(output_dir, 'fixed_img1_ori.jpg'),
                       pad_value=1)
            save_image(tensor=fixed_img2.cpu(),
                       fp=os.path.join(output_dir, 'fixed_img2_ori.jpg'),
                       pad_value=1)
            save_image(tensor=fixed_img3.cpu(),
                       fp=os.path.join(output_dir, 'fixed_img3_ori.jpg'),
                       pad_value=1)
            save_image(tensor=fixed_img4.cpu(),
                       fp=os.path.join(output_dir, 'fixed_img4_ori.jpg'),
                       pad_value=1)
            save_image(tensor=fixed_img5.cpu(),
                       fp=os.path.join(output_dir, 'fixed_img5_ori.jpg'),
                       pad_value=1)
            save_image(tensor=random_img.cpu(),
                       fp=os.path.join(output_dir, 'random_img_ori.jpg'),
                       pad_value=1)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)


    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()


    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'beta': self.win_beta,
                      'kld':self.win_kld,
                    #   'mu':self.win_mu,
                    #   'var':self.win_var,
                      }
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))


    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            # self.win_var = checkpoint['win_states']['var']
            # self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
