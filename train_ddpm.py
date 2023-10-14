import argparse
from models.ddpm import DDPM 
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
import os
import time
import numpy as np
from PIL import Image

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from models.data.dataloader import NpyDataModule

# This will save checkpoints to the './checkpoints' directory.
# The '{epoch}-{val_loss:.2f}' in the filename means that the saved model's name will contain the epoch and validation loss.


    
class SetupCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ckptdir= os.path.join(self.args.rootdir, "checkpoints")
        self.logdir = os.path.join(self.args.rootdir, "logs")
        self.resume = self.args.resume
        self.now = time.strftime("%Y-%m-%d-%H-%M-%S")

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            trainer.save_checkpoint(os.path.join(self.ckptdir, 'checkpoints'))

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)

            # if "callbacks" in self.lightning_config:
            #     if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
            #         os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            # print("Project config")
            # print(OmegaConf.to_yaml(self.config))
            # OmegaConf.save(self.config,
            #                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            # print("Lightning config")
            # print(OmegaConf.to_yaml(self.lightning_config))
            # OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
            #                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))
        else:
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass
    def on_train_start(self, trainer, pl_module):
        print("Training Arguments:")
        for arg, value in vars(self.args).items():
            print(f"{arg}:\t {value}")
        

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            TensorBoardLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, batch_output,  batch, batch_idx, dataloader_idx=1):
        pass
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        #     self.log_img(pl_module, batch, batch_idx, split="train")
        

    def on_validation_batch_end(self, trainer, pl_module, batch_output,  batch, batch_idx, dataloader_idx=1):
        # print("sjj is a big big big ben ben ben!")
        if batch_idx >= 1: #xsm: gansita
            return
        if not self.disabled and pl_module.global_step > 0:
            # print("I should log image here!")
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device)
        torch.cuda.synchronize(trainer.strategy.root_device)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_modul):
        torch.cuda.synchronize(trainer.strategy.root_device)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass




class VanillaDiffusionTrainer(nn.Module): #Main trainer
    def __init__(self, args):
        super(VanillaDiffusionTrainer, self).__init__()
        self.ddpm = DDPM.from_args(DDPM, args)
        self.data_module = NpyDataModule.from_args(NpyDataModule, args)
        cuda_callback = CUDACallback()
        img_logger_callback = ImageLogger(batch_frequency=args.img_batch_freq,  max_images=args.image_display_num, log_on_batch_idx=False, log_first_step=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath= os.path.join(args.rootdir, "checkpoints"),
            filename='{epoch}-{val_loss:.2f}',
            save_top_k=3,  # Save the top 3 models based on the metric defined in 'monitor'
            verbose=True,
            monitor='val/loss_simple',  # Choose the metric to monitor for model saving
            mode='min'  # 'min' indicates that the models with the smallest 'val_loss' values are saved
        )
        setup_callback = SetupCallback(args)
        self.trainer = pl.Trainer(callbacks=[setup_callback, img_logger_callback, cuda_callback, checkpoint_callback], 
                                  max_epochs=args.epochs, num_nodes=args.gpus)
    def train(self):
        self.trainer.fit(self.ddpm, self.data_module)
        
    def test(self):
        self.trainer.test(self.ddpm, self.data_module)
    
    def load(self, path):
        self.ddpm.load_state_dict(torch.load(path))
    
    def save(self, path):
        torch.save(self.ddpm.state_dict(), path)
    
    def sample(self, batch_size, T, device):
        return self.ddpm.sample(batch_size, T, device)
    
parser = argparse.ArgumentParser(description='Train a DDPM model')
parser.add_argument('--rootdir', type=str, default='.', help='root directory')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--img_batch_freq', type=int, default=100, help='image logging frequency')
parser.add_argument('--image_display_num', type=int, default=4, help='number of images to display')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

parser.add_argument('--timesteps', default=1000, type=int, help='Description')
parser.add_argument('--beta_schedule', default='linear', type=str, help='Description')
parser.add_argument('--loss_type', default='l2', type=str, help='Description')
parser.add_argument('--ckpt_path', default=None, type=str, help='Checkpoint path')
parser.add_argument('--ignore_keys', default=[], nargs='*', help='Keys to ignore')
parser.add_argument('--load_only_unet', default=False, action='store_true', help='Description')
parser.add_argument('--monitor', default='val/loss', type=str, help='Description')
parser.add_argument('--use_ema', default=True, action='store_true', help='Description')
parser.add_argument('--first_stage_key', default='image', type=str, help='Description')
parser.add_argument('--log_every_t', default=100, type=int, help='Description')
parser.add_argument('--clip_denoised', default=True, action='store_true', help='Description')
parser.add_argument('--linear_start', default=1e-4, type=float, help='Description')
parser.add_argument('--linear_end', default=2e-2, type=float, help='Description')
parser.add_argument('--cosine_s', default=8e-3, type=float, help='Description')
parser.add_argument('--given_betas', default=None, type=list, help='Description')
parser.add_argument('--original_elbo_weight', default=0., type=float, help='Description')
parser.add_argument('--v_posterior', default=0., type=float, help='Description')
parser.add_argument('--l_simple_weight', default=1., type=float, help='Description')
parser.add_argument('--parameterization', default='eps', type=str, help='Description')
parser.add_argument('--scheduler_config', default=None, type=list, help='Description')
parser.add_argument('--use_positional_encodings', default=False, action='store_true', help='Description')
parser.add_argument('--learn_logvar', default=False, action='store_true', help='Description')
parser.add_argument('--logvar_init', default=0., type=float, help='Description')
#Unet Config
parser.add_argument('--ch', type=int, default=224, help='Unet Channel size')
parser.add_argument('--out_ch', type=int, default=3, help='Output channel size')
parser.add_argument('--ch_mult', type=int, nargs='+', default=(1,2,4), help='Channel multipliers')
parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks') 
parser.add_argument('--attn_resolutions', type=int, nargs='+', default=(4,2), help='Attention resolutions')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--resamp_with_conv', type=bool, default=True, help='Resample with convolution')
parser.add_argument('--in_channels', type=int, default=20, help='Input channels')
parser.add_argument('--resolution', type=int, default=64, help='Resolution')
parser.add_argument('--use_timestep', type=bool, default=True, help='Use timestep')
parser.add_argument('--use_linear_attn', type=bool, default=False, help='Use linear attention')
parser.add_argument('--attn_type', type=str, default='vanilla', help='Attention type')
parser.add_argument('--use_mask', type=bool, default=True)
#data config
parser.add_argument('--data_path', type=str, default='data', help='Path to data')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--train_val_test_split', type=tuple, default=(0.8, 0.1, 0.1), help='Train, validation, and test split')

if __name__ == "__main__":
    args = parser.parse_args()
    print(torch.cuda.is_available())
    print("Cuda version is: ", torch.version.cuda)
    trainer = VanillaDiffusionTrainer(args)
    trainer.train()
    trainer.test()
    trainer.save(os.path.join(args.rootdir, "checkpoints", "model_final.ckpt"))
