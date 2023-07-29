#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import cv2
import torch.nn as nn
import torchvision.transforms.functional as tvF
from PIL import Image
from torch.optim import Adam, lr_scheduler
import tifffile as tiff

from unet import UNet2 as UNet
from utils import *

import os
import json


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()


    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print('Noise2Noise: Learning Image Restoration without Clean Data (Lethinen et al., 2018)')

        # Model (3x3=9 channels for Monte Carlo since it uses 3 HDR buffers)
        if self.p.noise_type == 'mc':
            self.is_mc = True
            self.model = UNet(in_channels=9)
        else:
            self.is_mc = False
            self.model = UNet(in_channels=1)

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.p.nb_epochs/4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss == 'hdr':
                assert self.is_mc, 'Using HDR loss on non Monte Carlo images'
                self.loss = HDRLoss()
            elif self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            if self.p.clean_targets:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-clean-%H%M}'
            else:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-%H%M}'
            if self.p.ckpt_overwrite:
                if self.p.clean_targets:
                    ckpt_dir_name = f'{self.p.noise_type}-clean'
                else:
                    ckpt_dir_name = self.p.noise_type

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, "stats")
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/n2n-{}.pt'.format(self.ckpt_dir, self.p.noise_type)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)


    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))


    def compute_sharpness(self, image):
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        mag = np.mean(np.abs(laplacian))

        return mag


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_list, valid_target_dir):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr, valid_sharp, valid_perc, valid_merged, valid_denoised, valid_target = self.eval(epoch, valid_list, valid_target_dir)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr, valid_sharp, valid_perc, valid_merged, valid_denoised, valid_target)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        stats['valid_sharpness'].append(valid_sharp)
        stats['valid_sharp_perc'].append(valid_perc)
        stats['valid_sharp_merged'].append(valid_merged)
        stats['valid_sharp_denoised'].append(valid_denoised)
        stats['valid_sharp_target'].append(valid_target)
        self.save_model(epoch, stats, epoch == 0)


        # Plot stats
        if self.p.plot_stats:
            loss_str = f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')
            plot_per_epoch(self.ckpt_dir, 'Valid sharpness difference', stats['valid_sharpness'], '')
            plot_per_epoch(self.ckpt_dir, 'Valid sharpness (merged)', stats['valid_sharp_merged'], '')
            plot_per_epoch(self.ckpt_dir, 'Valid sharpness (denoised)', stats['valid_sharp_denoised'], '')
            plot_per_epoch(self.ckpt_dir, 'Valid sharpness (target)', stats['valid_sharp_target'], '')
            plot_per_epoch(self.ckpt_dir, 'Valid sharpness diff (percentage)', stats['valid_sharp_perc'], '')


    def test(self, test_loader, show):
        """Evaluates denoiser on test set."""

        self.model.train(False)

        source_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        denoised_dir = os.path.dirname(self.p.data)
        save_path = os.path.join(denoised_dir, 'denoised')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for batch_idx, (source, target) in enumerate(test_loader):
            # Only do first <show> images
            if show == 0 or batch_idx >= show:
                break

            source_imgs.append(source)
            clean_imgs.append(target)

            if self.use_cuda:
                source = source.cuda()

            # Denoise
            denoised_img = self.model(source).detach()
            denoised_imgs.append(denoised_img)

        # Squeeze tensors
        source_imgs = [t.squeeze(0) for t in source_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]

        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        for i in range(len(source_imgs)):
            img_name = test_loader.dataset.imgs[i]
            create_montage(img_name, self.p.noise_type, save_path, source_imgs[i], denoised_imgs[i], clean_imgs[i], show)


    def merge_images(self, image1, image2, a):
        assert image1.size == image2.size, "size not equal!"

        image1 = image1.astype(np.float32)
        image2 = image2.astype(np.float32)

        merged_image = a * image1 + (1 - a) * image2
        merged_image = np.clip(merged_image, 0, 255)
        merged_image = merged_image.astype(np.uint8)

        return merged_image
    
    def add_gaussian_noise(self, image, mean=0, std_dev=10):
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image

    def eval(self, epoch, valid_list, valid_target_dir):
        """Evaluates denoiser on validation set."""

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()
        sharp_meter = AvgMeter()
        sharp_perc = AvgMeter()
        sharp_merged = AvgMeter()
        sharp_denoised = AvgMeter()
        sharp_target = AvgMeter()

        base = None

        for batch_idx, source_file in enumerate(valid_list):
            print("batch_idx:", batch_idx)
            #print(source_file)
            file_name = os.path.basename(source_file)
            file_prefix = file_name.split(".tif")[0]

            source = Image.open(source_file).convert("L")
            #print("source", source.size)
            if self.p.add_noise:
                base = tvF.to_tensor(source)
                source = np.asarray(source) 
                #noise = np.random.poisson(source)
                #noise_img = source + noise
                #source = 255 * (noise_img / np.amax(noise_img))
                source = self.add_gaussian_noise(source, 0, 0.5)
                source = Image.fromarray(source)
                
            source = tvF.to_tensor(source)
            #print("source", source.shape)
            target = source

            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

            # Denoise
            source_denoised = self.model(source)

            # Update loss
            loss = self.loss(source_denoised, target)
            loss_meter.update(loss.item())

            # old Compute PSRN
            if self.is_mc:
                source_denoised = reinhard_tonemap(source_denoised)
            source_denoised = source_denoised.cpu()
            target = target.cpu()
            #print("source_denoised, target", len(source_denoised), len(target), target.shape)
            #psnr_meter.update(psnr(source_denoised[0], target[0]).item())
            #psnr
            if self.p.add_noise:
                psnr_meter.update(psnr(source_denoised, base).item() - psnr(target, base).item())
            else:
                psnr_meter.update(-1)

            source_denoised = np.array(tvF.to_pil_image(source_denoised))
            target = np.array(tvF.to_pil_image(target))
            #merged
            merged_image = self.merge_images(target, source_denoised, 0.05)

            #save image
            tif_path = os.path.join(self.p.ckpt_save_path, "tif")
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(tif_path):
                os.mkdir(tif_path)
            if batch_idx == 0:
                tiff.imwrite(os.path.join(tif_path, file_prefix + "_epoch" + str(epoch) + "_denoised.tif"), source_denoised)
                tiff.imwrite(os.path.join(tif_path, file_prefix + "_epoch" + str(epoch) + "_merged.tif"), merged_image)
            if self.p.add_noise and batch_idx == 0:
                source_np = np.array(tvF.to_pil_image(source))
                tiff.imwrite(os.path.join(tif_path, file_prefix + "_epoch" + str(epoch) + "_noise.tif"), source_np)

            #update sharpness
            denoised_sharp = self.compute_sharpness(source_denoised) 
            merged_sharp = self.compute_sharpness(merged_image)
            target_sharp = self.compute_sharpness(target)
            sharp_diff = merged_sharp - target_sharp
            sharp_percentage = sharp_diff / target_sharp
            sharp_meter.update(sharp_diff)
            sharp_perc.update(sharp_percentage)
            sharp_merged.update(merged_sharp)
            sharp_denoised.update(denoised_sharp)
            sharp_target.update(target_sharp)

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg
        sharp_avg = sharp_meter.avg
        sharp_perc_avg = sharp_perc.avg
        sharp_merged_avg = sharp_merged.avg
        sharp_denoised_avg = sharp_denoised.avg
        sharp_target_avg = sharp_target.avg

        return valid_loss, valid_time, psnr_avg, sharp_avg, sharp_perc_avg, sharp_merged_avg, sharp_denoised_avg, sharp_target_avg


    def train(self, train_list, train_target_dir, valid_list, valid_target_dir):
        """Trains denoiser on training set."""

        self.model.train(True)

        self._print_params()
        num_batches = len(train_list)
        print(num_batches, self.p.report_interval)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches, {num_batches}, {self.p.report_interval}'

        # Dictionaries of tracked stats
        stats = {'noise_type': self.p.noise_type,
                 'noise_param': self.p.noise_param,
                 'train_loss': [],
                 'valid_loss': [],
                 'valid_sharpness': [],
                 'valid_sharp_perc': [],
                 'valid_sharp_merged': [],
                 'valid_sharp_denoised': [],
                 'valid_sharp_target': [],
                 'valid_psnr': []}

        # Main training loop
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            #for batch_idx, (source, target) in enumerate(train_loader):
            #/home/alyld7/data/1-whole_out/samp1/new_MidbrainvolumePamirNott090-9_org_samp1.tif
            for batch_idx, source_file in enumerate(train_list):
                print(source_file)
                target_file = os.path.basename(source_file)
                target_file = target_file.split("1.tif")[0]
                target_file = target_file + "2.tif"
                target_file = os.path.join(train_target_dir, target_file)

                target = Image.open(target_file).convert("L")
                target = tvF.to_tensor(target)

                source = Image.open(source_file).convert("L")
                source = tvF.to_tensor(source)
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                # Denoise image
                source_denoised = self.model(source)

                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_list, valid_target_dir)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))


class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""

        super(HDRLoss, self).__init__()
        self._eps = eps


    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""

        loss = ((denoised - target) ** 2) / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))
