"""solver.py"""

import os
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from utils import DataGather, mkdirs, grid2gif
from ops import recon_loss, kl_divergence, permute_dims, \
    cls_loss, permute_sens_dims
from model import VAE, Discriminator, FFVAE
from dataset import return_data, return_train_data, return_val_data
from saliency.fullgrad import FullGrad


class Solver(object):
    def __init__(self, args):
        # Misc
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.name = args.name
        self.max_iter = int(args.max_iter)
        self.print_iter = args.print_iter
        self.global_iter = 0
        self.test_iter = 0
        self.pbar = tqdm(total=self.max_iter)

        self.model = args.model

        # Data
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        # Hyperparameters
        self.z_dim = args.z_dim
        self.gamma = args.gamma
        self.beta = args.beta
        self.alpha = args.alpha
        self.phi = args.phi
        self.eta1 = args.eta1
        self.eta2 = args.eta2

        self.n_sens = args.n_sens
        self.sens_idx = args.sens_idx

        self.ot_idx = args.ot_idx
        self.sens_cls_name = args.sens_cls_name
        self.sens_ckpt = args.sens_ckpt

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D

        # Models & Optimizers
        if self.model == 'vanilla' or self.model == 'beta':
            self.VAE = VAE(self.z_dim).to(self.device)
            self.nc = 3
            self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                        betas=(self.beta1_VAE, self.beta2_VAE))

            self.nets = [self.VAE]
        elif self.model == 'factor':
            self.VAE = VAE(self.z_dim).to(self.device)
            self.nc = 3
            self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                        betas=(self.beta1_VAE, self.beta2_VAE))

            self.D = Discriminator(self.z_dim).to(self.device)
            self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                      betas=(self.beta1_D, self.beta2_D))

            self.nets = [self.VAE, self.D]
        elif self.model == 'ff':
            self.VAE = FFVAE(self.z_dim, self.n_sens).to(self.device)
            self.nc = 3
            self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                        betas=(self.beta1_VAE, self.beta2_VAE))

            self.D = Discriminator(self.z_dim).to(self.device)
            self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                      betas=(self.beta1_D, self.beta2_D))

            self.nets = [self.VAE, self.D]
        elif self.model == 'sens_cls' or self.model == 'ot_bl_cls' or \
                self.model == 'ot_cls':
            self.train_data_loader = return_train_data(args)
            self.val_data_loader = return_val_data(args)

            self.D = Discriminator(self.z_dim).to(self.device)
            self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                      betas=(self.beta1_D, self.beta2_D))

            self.nets = [self.D]
        else:
            pass

        self.image_gather = DataGather('true', 'recon')

        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        self.ckpt_save_iter = args.ckpt_save_iter
        mkdirs(self.ckpt_dir)
        if args.ckpt_load:
            self.load_checkpoint(args.ckpt_load)

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(args.output_dir, args.name)
        self.output_save = args.output_save
        mkdirs(self.output_dir)

    def train_vanilla(self):
        """
        Train vanilla VAE.
        Refer to the original paper for details
        at https://arxiv.org/pdf/1312.6114.pdf.
        """
        self.net_mode(train=True)

        out = False
        while not out:
            for x_true1, y_1, x_true2, y_2 in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kld = kl_divergence(mu, logvar)

                vae_loss = vae_recon_loss + vae_kld

                self.optim_VAE.zero_grad()
                vae_loss.backward()
                self.optim_VAE.step()

                if self.global_iter % self.print_iter == 0:
                    self.pbar.write(('[{}] vae_recon_loss:{:.3f} '
                                     'vae_kld:{:.3f}').format(
                        self.global_iter, vae_recon_loss.item(),
                        vae_kld.item()))

                if self.global_iter % self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)

                if self.global_iter % 10000 == 0:
                    self.image_gather.insert(
                        true=x_true1.data.cpu(),
                        recon=F.sigmoid(x_recon).data.cpu())
                    self.visualize_recon()
                    self.image_gather.flush()

                if self.global_iter % 10000 == 0:
                    if self.dataset.lower() == '3dchairs':
                        self.visualize_traverse(limit=2, inter=0.5)
                    else:
                        self.visualize_traverse(limit=3, inter=2/3)

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        self.pbar.write("[Training Finished]")
        self.pbar.close()

    def train_beta(self):
        """
        Train beta-VAE.
        Refer to the original paper for details
        at https://openreview.net/pdf?id=Sy2fzU9gl.
        """
        self.net_mode(train=True)

        out = False
        while not out:
            for x_true1, y_1, x_true2, y_2 in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kld = kl_divergence(mu, logvar)

                vae_loss = vae_recon_loss + self.beta * vae_kld

                self.optim_VAE.zero_grad()
                vae_loss.backward()
                self.optim_VAE.step()

                if self.global_iter % self.print_iter == 0:
                    self.pbar.write(('[{}] vae_recon_loss:{:.3f} '
                                     'vae_kld:{:.3f}').format(
                        self.global_iter, vae_recon_loss.item(),
                        vae_kld.item()))

                if self.global_iter % self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)

                if self.global_iter % 10000 == 0:
                    self.image_gather.insert(
                        true=x_true1.data.cpu(),
                        recon=F.sigmoid(x_recon).data.cpu())
                    self.visualize_recon()
                    self.image_gather.flush()

                if self.global_iter % 10000 == 0:
                    if self.dataset.lower() == '3dchairs':
                        self.visualize_traverse(limit=2, inter=0.5)
                    else:
                        self.visualize_traverse(limit=3, inter=2/3)

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        self.pbar.write("[Training Finished]")
        self.pbar.close()

    def train_factor(self):
        """
        Train FactorVAE.
        Refer to the original paper for details
        at http://proceedings.mlr.press/v80/kim18b/kim18b.pdf.
        """
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long,
                          device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long,
                            device=self.device)

        out = False
        while not out:
            for x_true1, y_1, x_true2, y_2 in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kld = kl_divergence(mu, logvar)

                D_z = self.D(z)
                vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

                vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss

                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_VAE.step()

                x_true2 = x_true2.to(self.device)
                z_prime = self.VAE(x_true2, no_dec=True)
                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = self.D(z_pperm)
                D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) +
                                   F.cross_entropy(D_z_pperm, ones))

                self.optim_D.zero_grad()
                D_tc_loss.backward()
                self.optim_D.step()

                if self.global_iter % self.print_iter == 0:
                    self.pbar.write(('[{}] vae_recon_loss:{:.3f} '
                                     'vae_kld:{:.3f} vae_tc_loss:{:.3f} '
                                     'D_tc_loss:{:.3f}').format(
                        self.global_iter, vae_recon_loss.item(),
                        vae_kld.item(), vae_tc_loss.item(), D_tc_loss.item()))

                if self.global_iter % self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)

                if self.global_iter % 10000 == 0:
                    self.image_gather.insert(
                        true=x_true1.data.cpu(),
                        recon=F.sigmoid(x_recon).data.cpu())
                    self.visualize_recon()
                    self.image_gather.flush()

                if self.global_iter % 10000 == 0:
                    if self.dataset.lower() == '3dchairs':
                        self.visualize_traverse(limit=2, inter=0.5)
                    else:
                        self.visualize_traverse(limit=3, inter=2/3)

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        self.pbar.write("[Training Finished]")
        self.pbar.close()

    def train_ff(self):
        """
        Train FFVAE.
        Refer to the original paper for details
        at http://proceedings.mlr.press/v97/creager19a/creager19a.pdf.
        """
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long,
                          device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long,
                            device=self.device)

        out = False
        while not out:
            for x_true1, y_1, x_true2, y_2 in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                y_1 = y_1.to(self.device)

                x_recon, mu_n, logvar_n, zb, b_logits = self.VAE(x_true1)

                if self.n_sens != 1:
                    b_logits = b_logits.squeeze(1)

                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kld = kl_divergence(mu_n, logvar_n)

                D_zb = self.D(zb)
                vae_tc_loss = (D_zb[:, :1] - D_zb[:, 1:]).mean()

                vae_cls_loss = cls_loss(y_1[:, self.sens_idx], b_logits)

                vae_loss = vae_recon_loss + vae_kld + \
                    self.gamma*vae_tc_loss + self.alpha*vae_cls_loss

                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_VAE.step()

                x_true2 = x_true2.to(self.device)
                zb_prime = self.VAE(x_true2, no_dec=True)
                zb_pperm = permute_sens_dims(zb_prime, self.n_sens).detach()
                D_zb_pperm = self.D(zb_pperm)
                D_tc_loss = 0.5 * (F.cross_entropy(D_zb, zeros) +
                                   F.cross_entropy(D_zb_pperm, ones))

                self.optim_D.zero_grad()
                D_tc_loss.backward()
                self.optim_D.step()

                if self.global_iter % self.print_iter == 0:
                    self.pbar.write(('[{}] vae_recon_loss:{:.3f} '
                                     'vae_kld:{:.3f} vae_tc_loss:{:.3f} '
                                     'vae_cls_loss:{:.3f} '
                                     'D_tc_loss:{:.3f}').format(
                        self.global_iter, vae_recon_loss.item(),
                        vae_kld.item(), vae_tc_loss.item(),
                        vae_cls_loss.item(), D_tc_loss.item()))

                if self.global_iter % self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)

                if self.global_iter % 10000 == 0:
                    self.image_gather.insert(
                        true=x_true1.data.cpu(),
                        recon=F.sigmoid(x_recon).data.cpu())
                    self.visualize_recon()
                    self.image_gather.flush()

                if self.global_iter % 10000 == 0:
                    if self.dataset.lower() == '3dchairs':
                        self.visualize_traverse(limit=2, inter=0.5)
                    else:
                        self.visualize_traverse(limit=3, inter=2/3)

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        self.pbar.write("[Training Finished]")
        self.pbar.close()

    def test_vae(self):
        """
        Test VAE by feeding it with inputs to generate latent codes.
        """
        self.max_iter = len(self.data_loader)
        self.pbar = tqdm(total=self.max_iter)
        self.net_mode(train=False)

        output_dir = os.path.join(self.output_dir, 'z_and_y')
        mkdirs(output_dir)

        if self.batch_size != 1:
            raise Exception('Batch size is not 1. '
                            'To save latent code z, '
                            'batch size must be 1.')

        out = False
        while not out:
            for x_true1, y_1, x_true2, y_2 in self.data_loader:
                self.test_iter += 1
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                y_1 = y_1.to(self.device)
                z = self.VAE(x_true1, no_dec=True)

                y_save = y_1.detach().squeeze().cpu().numpy()
                z_save = z.detach().cpu().numpy()

                np.savez('{0}/{1:06d}.npz'.format(output_dir, self.test_iter),
                         z=z_save, y=y_save)

                if self.test_iter >= self.max_iter:
                    out = True
                    break

        self.pbar.write("[Test Finished]")
        self.pbar.close()

    def train_sens_cls(self):
        """
        Train sensitive classifier with latent codes.
        """
        self.net_mode(train=True)
        num_epochs = 120

        print(f"\n[Training Begins. Total Epochs: {num_epochs}]")

        for epoch in range(1, num_epochs+1):
            training_loss = 0.0

            sum_correct = 0
            sum_num = 0

            with tqdm(self.train_data_loader, unit="batch") as pbar:
                for z, y in pbar:
                    pbar.set_description(f"Epoch {epoch}")
                    self.global_iter += 1

                    z = z.to(self.device)
                    y = y.to(self.device)

                    D_z = self.D(z)

                    if self.n_sens == 1:
                        sens_cls_loss = F.cross_entropy(
                            D_z, y[:, self.sens_idx].long().squeeze())
                    else:
                        labels = ~((y[:, [self.sens_idx[0]]] == 0) &
                                   (y[:, [self.sens_idx[1]]] == 1))
                        sens_cls_loss = F.cross_entropy(
                            D_z, labels.long().squeeze())

                    self.optim_D.zero_grad()
                    sens_cls_loss.backward()
                    self.optim_D.step()

                    training_loss += sens_cls_loss.item()

                    if len(D_z.size()) == 1:
                        results = \
                            (D_z.data.max(0, keepdim=True)[1]).unsqueeze(0)
                    else:
                        results = D_z.data.max(1, keepdim=True)[1]

                    if self.n_sens == 1:
                        ground_truths = y[:, self.sens_idx].long()
                    else:
                        ground_truths = labels.long()

                    n_correct = torch.sum(torch.eq(results, ground_truths))
                    sum_correct += n_correct.item()
                    sum_num += ground_truths.size(0)

            self.save_checkpoint(epoch)
            print(('\n epoch [{}] training_loss:{:.3f} '
                   'training_accuracy:{:.3f}').format(
                       epoch, training_loss,
                       float(sum_correct) / float(sum_num)))
            self.val_sens_cls()

        print("\n[Training Finished]")

    def val_sens_cls(self):
        """
        Evaluate the accuracy of sensitive classifier
        """
        self.net_mode(train=False)

        sum_correct = 0
        sum_num = 0

        print("\n[Validation Begins]")

        with tqdm(self.val_data_loader, unit="batch") as pbar:
            for z, y in pbar:
                pbar.set_description("Validation")

                z = z.to(self.device)
                y = y.to(self.device)

                D_z = self.D(z)

                if len(D_z.size()) == 1:
                    D_z = F.softmax(D_z, 0).data
                    results = \
                        (D_z.max(0, keepdim=True)[1]).unsqueeze(0)
                else:
                    D_z = F.softmax(D_z, 1).data
                    results = D_z.max(1, keepdim=True)[1]

                if self.n_sens == 1:
                    ground_truths = y[:, self.sens_idx].long()
                else:
                    labels = ~((y[:, [self.sens_idx[0]]] == 0) &
                               (y[:, [self.sens_idx[1]]] == 1))
                    ground_truths = labels.long()

                n_correct = torch.sum(torch.eq(results, ground_truths))
                sum_correct += n_correct.item()
                sum_num += ground_truths.size(0)

        print('\n val_accuracy:{:.3f}'.format(
            float(sum_correct) / float(sum_num)))
        print("\n[Validation Finished]")

        self.net_mode(train=True)

    def test_sens_cls(self):
        """
        Test sensitive classifiers
        """
        print('\n[Test Begins]')

        self.net_mode(train=False)

        sum_correct = 0
        sum_num = 0

        for z, y in self.val_data_loader:
            z = z.to(self.device)
            y = y.to(self.device)

            D_z = self.D(z)

            if len(D_z.size()) == 1:
                D_z = F.softmax(D_z, 0).data
                results = \
                    (D_z.max(0, keepdim=True)[1]).unsqueeze(0)
            else:
                D_z = F.softmax(D_z, 1).data
                results = D_z.max(1, keepdim=True)[1]

            if self.n_sens == 1:
                ground_truths = y[:, self.sens_idx].long()
            else:
                labels = ~((y[:, [self.sens_idx[0]]] == 0) &
                           (y[:, [self.sens_idx[1]]] == 1))
                ground_truths = labels.long()

            n_correct = torch.sum(torch.eq(results, ground_truths))
            sum_correct += n_correct.item()
            sum_num += ground_truths.size(0)

        print('\n val_accuracy:{:.3f}'.format(
            float(sum_correct) / float(sum_num)))

        print("[Test Finished]")

    def train_ot_bl_cls(self):
        """
        Train and debias downstream task classifier by removing sensitive
        dimensions according to previous methods.
        """
        self.net_mode(train=True)
        num_epochs = 100

        print(f"\n[Training Begins. Total Epochs: {num_epochs}]")

        self.val_ot_bl_cls()

        for epoch in range(1, num_epochs+1):
            training_loss = 0.0

            sum_correct = 0
            sum_num = 0

            for z, y in self.train_data_loader:
                self.global_iter += 1

                z = z.to(self.device)
                y = y.to(self.device)

                # remove "sensitive dimension(s)"
                # Note: sensitive dimension index or indices
                # need to be specified here.
                if 'ff_celeba' in self.dataset:
                    z[:, self.z_dim - self.n_sens:] = 0.0
                elif 'factor_celeba' in self.dataset:
                    z[:, 0] = 0.0
                elif 'vanilla_celeba' in self.dataset:
                    z[:, 7] = 0.0
                elif 'beta_celeba' in self.dataset:
                    z[:, 7] = 0.0

                D_z = self.D(z)

                ot_cls_loss = F.cross_entropy(
                    D_z, y[:, [self.ot_idx]].long().squeeze())

                self.optim_D.zero_grad()
                ot_cls_loss.backward()
                self.optim_D.step()

                training_loss += ot_cls_loss.item()

                if len(D_z.size()) == 1:
                    results = \
                        (D_z.data.max(0, keepdim=True)[1]).unsqueeze(0)
                else:
                    results = D_z.data.max(1, keepdim=True)[1]

                ground_truths = y[:, [self.ot_idx]].long()
                n_correct = torch.sum(torch.eq(results, ground_truths))
                sum_correct += n_correct.item()
                sum_num += ground_truths.size(0)

                if self.global_iter % 703 == 0:
                    self.val_ot_bl_cls()

            self.save_checkpoint(epoch)
            print(('\n epoch [{}] training_loss:{:.3f} '
                   'training_accuracy:{:.3f}').format(
                       epoch, training_loss,
                       float(sum_correct) / float(sum_num)))

        print("\n[Training Finished]")

    def val_ot_bl_cls(self):
        """
        Evaluate downstream task classifier trained by
        previous debiasing methods.
        """
        self.net_mode(train=False)

        sum_correct = 0
        sum_num = 0

        sum_pos_a = 0
        sum_tp_a = 0
        sum_cor_a = 0
        sum_num_a = 0

        sum_pos_b = 0
        sum_tp_b = 0
        sum_cor_b = 0
        sum_num_b = 0

        print("\n[Validation Begins]")

        for z, y in self.val_data_loader:
            z = z.to(self.device)
            y = y.to(self.device)

            # remove "sensitive dimension(s)"
            # Note: sensitive dimension index or indices
            # need to be specified here.
            if 'ff_celeba' in self.dataset:
                z[:, self.z_dim - self.n_sens:] = 0.0
            elif 'factor_celeba' in self.dataset:
                z[:, 0] = 0.0
            elif 'vanilla_celeba' in self.dataset:
                z[:, 7] = 0.0
            elif 'beta_celeba' in self.dataset:
                z[:, 7] = 0.0

            D_z = self.D(z)

            if len(D_z.size()) == 1:
                D_z = F.softmax(D_z, 0).data
                results = \
                    (D_z.max(0, keepdim=True)[1]).unsqueeze(0)
            else:
                D_z = F.softmax(D_z, 1).data
                results = D_z.max(1, keepdim=True)[1]

            ground_truths = y[:, [self.ot_idx]].long()
            n_correct = torch.sum(torch.eq(results, ground_truths))
            sum_correct += n_correct.item()
            sum_num += ground_truths.size(0)

            if self.n_sens == 1:
                sens_gt = y[:, self.sens_idx].long()
            else:
                labels = ~((y[:, [self.sens_idx[0]]] == 0) &
                           (y[:, [self.sens_idx[1]]] == 1))
                sens_gt = labels.long()

            a = (sens_gt == 0)
            b = (sens_gt == 1)
            pos = (results == 1)
            tp = (ground_truths == 1)
            cor = torch.eq(results, ground_truths)

            pos_a = a & pos
            pos_b = b & pos

            tp_a = a & pos & tp
            tp_b = b & pos & tp

            cor_a = a & cor
            cor_b = b & cor

            sum_pos_a += pos_a.sum().item()
            sum_pos_b += pos_b.sum().item()

            sum_tp_a += tp_a.sum().item()
            sum_tp_b += tp_b.sum().item()

            sum_cor_a += cor_a.sum().item()
            sum_cor_b += cor_b.sum().item()

            sum_num_a += a.sum().item()
            sum_num_b += b.sum().item()

        dp = abs(float(sum_pos_a)/float(sum_num_a) -
                 float(sum_pos_b)/float(sum_num_b))

        eo = abs(float(sum_tp_a)/float(sum_num_a) -
                 float(sum_tp_b)/float(sum_num_b))

        ap = abs(float(sum_cor_a)/float(sum_num_a) -
                 float(sum_cor_b)/float(sum_num_b))

        print('\n val_accuracy:{:.3f}'.format(
            float(sum_correct) / float(sum_num)))
        print(' demographic parity:{:.3f}'.format(dp))
        print(' equal opportunity:{:.3f}'.format(eo))
        print(' accuracy parity:{:.3f}'.format(ap))
        print(" [Validation Finished]")

        self.net_mode(train=True)

    def test_ot_bl_cls(self):
        """
        Test downstream task classifier trained by
        previous debiasing methods.
        """
        self.val_ot_bl_cls()

    def train_ot_cls(self):
        """
        Train and debias downstream task classifier using DVGE.
        """
        self.sens_cls = Discriminator(self.z_dim).to(self.device)
        ckpt_path = os.path.join('checkpoints', self.sens_cls_name,
                                 self.sens_ckpt)

        if os.path.isfile(ckpt_path):
            with open(ckpt_path, 'rb') as f:
                sens_cls_ckpt = torch.load(f)

            self.sens_cls.load_state_dict(sens_cls_ckpt['model_states']['D'])
            self.sens_cls.eval()

            print(f'\n[sens_cls checkpoint loaded from {ckpt_path}]')
        else:
            print('\n[No sens_cls checkpoint found]')

        # initialize FullGrad instances for sensitive focus and
        # downstream task focus
        self.sens_fg = FullGrad(self.sens_cls, im_size=(10,))
        self.ot_fg = FullGrad(self.D, im_size=(10,))

        self.net_mode(train=True)
        num_epochs = 100

        print(f"\n[Training Begins. Total Epochs: {num_epochs}]")

        self.val_ot_cls()

        for epoch in range(1, num_epochs+1):
            training_loss = 0.0

            sum_correct = 0
            sum_num = 0

            for z, y in self.train_data_loader:
                self.global_iter += 1

                z = z.to(self.device)
                y = y.to(self.device)

                # generate sensitive focus
                sens_sm = self.sens_fg.saliency(z).data
                # generate downstream task focus
                ot_sm = self.ot_fg.saliency(z).data

                # perform bidirectional perturbation
                new_z = z + self.eta1 * sens_sm - self.eta2 * ot_sm

                self.net_mode(train=True)

                D_z = self.D(new_z)
                cls_loss = F.cross_entropy(
                    D_z, y[:, [self.ot_idx]].long().squeeze())

                self.optim_D.zero_grad()
                cls_loss.backward()
                self.optim_D.step()

                training_loss += cls_loss.item()

                if len(D_z.size()) == 1:
                    D_z = F.softmax(D_z, 0).data
                    results = \
                        (D_z.max(0, keepdim=True)[1]).unsqueeze(0)
                else:
                    D_z = F.softmax(D_z, 1).data
                    results = D_z.max(1, keepdim=True)[1]

                ground_truths = y[:, [self.ot_idx]].long()
                n_correct = torch.sum(torch.eq(results, ground_truths))
                sum_correct += n_correct.item()
                sum_num += ground_truths.size(0)

                if self.global_iter % 703 == 0:
                    self.val_ot_cls()

            self.save_checkpoint(epoch)
            print(('\n epoch [{}] training_loss:{:.3f} '
                   'training_accuracy:{:.3f}').format(
                       epoch, training_loss,
                       float(sum_correct) / float(sum_num)))

        print("\n[Training Finished]")

    def val_ot_cls(self):
        """
        Evaluate downstream task classifier using DVGE
        without performing bidirectional perturbation
        """
        self.net_mode(train=False)

        sum_correct = 0
        sum_num = 0

        sum_pos_a = 0
        sum_tp_a = 0
        sum_cor_a = 0
        sum_num_a = 0

        sum_pos_b = 0
        sum_tp_b = 0
        sum_cor_b = 0
        sum_num_b = 0

        print("\n[Validation Begins]")

        for z, y in self.val_data_loader:
            z = z.to(self.device)
            y = y.to(self.device)

            D_z = self.D(z)

            if len(D_z.size()) == 1:
                D_z = F.softmax(D_z, 0).data
                results = \
                    (D_z.max(0, keepdim=True)[1]).unsqueeze(0)
            else:
                D_z = F.softmax(D_z, 1).data
                results = D_z.max(1, keepdim=True)[1]

            ground_truths = y[:, [self.ot_idx]].long()
            n_correct = torch.sum(torch.eq(results, ground_truths))
            sum_correct += n_correct.item()
            sum_num += ground_truths.size(0)

            if self.n_sens == 1:
                sens_gt = y[:, self.sens_idx].long()
            else:
                labels = ~((y[:, [self.sens_idx[0]]] == 0) &
                           (y[:, [self.sens_idx[1]]] == 1))
                sens_gt = labels.long()

            a = (sens_gt == 0)
            b = (sens_gt == 1)
            pos = (results == 1)
            tp = (ground_truths == 1)
            cor = torch.eq(results, ground_truths)

            pos_a = a & pos
            pos_b = b & pos

            tp_a = a & pos & tp
            tp_b = b & pos & tp

            cor_a = a & cor
            cor_b = b & cor

            sum_pos_a += pos_a.sum().item()
            sum_pos_b += pos_b.sum().item()

            sum_tp_a += tp_a.sum().item()
            sum_tp_b += tp_b.sum().item()

            sum_cor_a += cor_a.sum().item()
            sum_cor_b += cor_b.sum().item()

            sum_num_a += a.sum().item()
            sum_num_b += b.sum().item()

        dp = abs(float(sum_pos_a)/float(sum_num_a) -
                 float(sum_pos_b)/float(sum_num_b))

        eo = abs(float(sum_tp_a)/float(sum_num_a) -
                 float(sum_tp_b)/float(sum_num_b))

        ap = abs(float(sum_cor_a)/float(sum_num_a) -
                 float(sum_cor_b)/float(sum_num_b))

        print('\n val_accuracy:{:.3f}'.format(
            float(sum_correct) / float(sum_num)))
        print(' demographic parity:{:.3f}'.format(dp))
        print(' equal opportunity:{:.3f}'.format(eo))
        print(' accuracy parity:{:.3f}'.format(ap))
        print("\n[Validation Finished]")

        self.net_mode(train=True)

    def test_ot_cls(self):
        """
        Test downstream task classifier using DVGE
        without performing bidirectional perturbation
        """
        self.val_ot_cls()

    def visualize_recon(self):
        """
        Visualize reconstructed images by VAE
        """
        data = self.image_gather.data
        true_image = data['true'][0]
        recon_image = data['recon'][0]

        true_image = make_grid(true_image)
        recon_image = make_grid(recon_image)
        sample = torch.stack([true_image, recon_image], dim=0)

        output_dir = os.path.join(self.output_dir, str(self.global_iter))
        mkdirs(output_dir)

        save_image(tensor=sample,
                   filename=os.path.join(output_dir, 'recon.jpg'),
                   nrow=2)

    def visualize_traverse(self, limit=3, inter=2/3, loc=-1):
        """
        Visualize reconstructed images by tranvesing values of
        latent code dimensions
        """
        self.net_mode(train=False)

        decoder = self.VAE.decode
        encoder = self.VAE.encode
        interpolation = torch.arange(-limit, limit+0.1, inter)

        random_img = self.data_loader.dataset.__getitem__(0)[2]
        random_img = random_img.to(self.device).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        if self.dataset.lower() == 'dsprites':
            fixed_idx1 = 87040  # square
            fixed_idx2 = 332800  # ellipse
            fixed_idx3 = 578560  # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square': fixed_img_z1, 'fixed_ellipse': fixed_img_z2,
                 'fixed_heart': fixed_img_z3, 'random_img': random_img_z}

        elif self.dataset.lower() == 'celeba':
            fixed_idx1 = 191281  # 'CelebA/img_align_celeba/191282.jpg'
            fixed_idx2 = 143307  # 'CelebA/img_align_celeba/143308.jpg'
            fixed_idx3 = 101535  # 'CelebA/img_align_celeba/101536.jpg'
            fixed_idx4 = 70059  # 'CelebA/img_align_celeba/070060.jpg'

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = encoder(fixed_img4)[:, :self.z_dim]

            Z = {'fixed_1': fixed_img_z1, 'fixed_2': fixed_img_z2,
                 'fixed_3': fixed_img_z3, 'fixed_4': fixed_img_z4,
                 'random': random_img_z}

        elif self.dataset.lower() == '3dchairs':
            # 3DChairs/images/4682_image_052_p030_t232_r096.png
            fixed_idx1 = 40919
            # 3DChairs/images/14657_image_020_p020_t232_r096.png
            fixed_idx2 = 5172
            # 3DChairs/images/30099_image_052_p030_t232_r096.png
            fixed_idx3 = 22330

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_1': fixed_img_z1, 'fixed_2': fixed_img_z2,
                 'fixed_3': fixed_img_z3, 'random': random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)[0]
            fixed_img = fixed_img.to(self.device).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)

            Z = {'fixed_img': fixed_img_z, 'random_img': random_img_z,
                 'random_z': random_z}

        gifs = []
        for key in Z:
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()

        if self.output_save:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            mkdirs(output_dir)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation),
                             self.nc, 64, 64).transpose(1, 2)

            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(
                        tensor=gifs[i][j].cpu(),
                        filename=os.path.join(output_dir,
                                              '{}_{}.jpg'.format(key, j)),
                        nrow=self.z_dim, pad_value=1)

                grid2gif(str(os.path.join(output_dir, key+'*.jpg')),
                         str(os.path.join(output_dir, key+'.gif')), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='last', verbose=True):
        if self.model == 'vanilla' or self.model == 'beta':
            model_states = {'VAE': self.VAE.state_dict()}
            optim_states = {'optim_VAE': self.optim_VAE.state_dict()}
        elif self.model == 'factorS':
            model_states = {'D1': self.D1.state_dict(),
                            'D2': self.D2.state_dict(),
                            'VAE': self.VAE.state_dict()}
            optim_states = {'optim_D1': self.optim_D1.state_dict(),
                            'optim_D2': self.optim_D2.state_dict(),
                            'optim_VAE': self.optim_VAE.state_dict()}
        elif self.model == 'sens_cls' or self.model == 'ot_bl_cls' or \
                self.model == 'ot_cls':
            model_states = {'D': self.D.state_dict()}
            optim_states = {'optim_D': self.optim_D.state_dict()}
        else:
            model_states = {'D': self.D.state_dict(),
                            'VAE': self.VAE.state_dict()}
            optim_states = {'optim_D': self.optim_D.state_dict(),
                            'optim_VAE': self.optim_VAE.state_dict()}

        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(
                filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']

            if self.model == 'vanilla' or self.model == 'beta':
                self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
                self.optim_VAE.load_state_dict(
                    checkpoint['optim_states']['optim_VAE'])
            elif self.model == 'sens_cls' or self.model == 'ot_bl_cls' or \
                    self.model == 'ot_cls':
                self.D.load_state_dict(checkpoint['model_states']['D'])
                self.optim_D.load_state_dict(
                    checkpoint['optim_states']['optim_D'])
            else:
                self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
                self.optim_VAE.load_state_dict(
                    checkpoint['optim_states']['optim_VAE'])

                self.D.load_state_dict(checkpoint['model_states']['D'])
                self.optim_D.load_state_dict(
                    checkpoint['optim_states']['optim_D'])

            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(
                    filepath, self.global_iter))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(
                    filepath))
