"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid

from util import get_unet_from_args

from ldm.util import exists, default, count_params
from ldm.modules.ema import LitEma
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule): #xsm: modified for masking
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=64,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 lr = 1e-4,
                 use_mask = True
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.use_mask = use_mask
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = get_unet_from_args(unet_config)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        self.lr = lr

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
    
    def from_args(cls, args):
        return cls(unet_config=args,
                   timesteps=args.timesteps,
                   beta_schedule=args.beta_schedule,
                   loss_type=args.loss_type,
                   ckpt_path=args.ckpt_path,
                   ignore_keys=args.ignore_keys,
                   load_only_unet=args.load_only_unet,
                   monitor=args.monitor,
                   use_ema=args.use_ema,
                   first_stage_key=args.first_stage_key,
                   image_size=args.resolution,
                   channels=args.out_ch,
                   log_every_t=args.log_every_t,
                   clip_denoised=args.clip_denoised,
                   linear_start=args.linear_start,
                   linear_end=args.linear_end,
                   cosine_s=args.cosine_s,
                   given_betas=args.given_betas,
                   original_elbo_weight=args.original_elbo_weight,
                   v_posterior=args.v_posterior,
                   l_simple_weight=args.l_simple_weight,
                   parameterization=args.parameterization,
                   scheduler_config=args.scheduler_config,
                   use_positional_encodings=args.use_positional_encodings,
                   learn_logvar=args.learn_logvar,
                   logvar_init=args.logvar_init,
                   lr = args.lr,
                   use_mask = args.use_mask,
                   )


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, mask, clip_denoised: bool):
        model_out = self.model(x, t, cond, mask)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon[:,0:1,...].clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, mask, clip_denoised=True, repeat_noise=False):
        x = x[:, 0:1, ...]
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t,cond=cond, mask=mask, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        model_mean[:,0:1,...]=model_mean[:,0:1,...] + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return model_mean

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, mask=None, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        if mask is None:
            mask = torch.zeros((b, 1, self.image_size, self.image_size), device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond, mask,
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, mask, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), cond, mask, 
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
    def get_loss_masked(self, pred_img, pred_sunny, pred_inval, target_img, target_sunny, target_inval, 
                        alpha_sunny, alpha_inval, mean=True):
        masking = (1-target_sunny) * (1-target_inval)
        target_img_masked = target_img * masking
        pred_img_masked = pred_img * masking
        mask_ratio = masking.sum() / masking.numel()
        
        
        if self.loss_type == 'l1':
            loss = (target_img_masked - pred_img_masked).abs() / mask_ratio + alpha_sunny * (target_sunny - pred_sunny).abs() + \
            alpha_inval * (target_inval - pred_inval).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target_img_masked, pred_img_masked) / mask_ratio + alpha_sunny * \
                torch.nn.functional.binary_cross_entropy(torch.sigmoid(pred_sunny), target_sunny) + alpha_inval * \
                torch.nn.functional.binary_cross_entropy(torch.sigmoid(pred_inval), target_inval)
            else:
                loss = torch.nn.functional.mse_loss(target_img_masked, pred_img_masked, reduction='none') / mask_ratio + alpha_sunny * \
                torch.nn.functional.binary_cross_entropy(torch.sigmoid(pred_sunny), target_sunny, reduction='none') + alpha_inval * \
                torch.nn.functional.binary_cross_entropy(torch.sigmoid(pred_inval), target_inval, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return loss
    
    def p_losses_masked(self, x_start, t, noise=None, cond=None, mask=None, alpha_sunny=0.1, alpha_inval=0.1):
        assert mask is not None, "If you don't want to use mask, turn off the use_mask option."
        #check if mask has the required keys
        assert 'sunny_mask' in mask.keys() and 'invalid_mask' in mask.keys() and 'ori_invalid_mask' in mask.keys(), \
            "Mask should have keys: sunny_mask, invalid_mask, ori_invalid_mask"
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        out = self.model(x_noisy, t, cond, mask['invalid_mask'])
        assert len(out.shape) == 4, "Output should be 4D tensor even if batch_size==1"
        img_out, sunny_out, inval_out = torch.split(out, 1, 1)
        
        img_out = img_out.squeeze(1)
        sunny_out = sunny_out.squeeze(1)
        inval_out = inval_out.squeeze(1)
        # print("img_out.shape==", img_out.shape)
        # print("sunny_out.shape==", sunny_out.shape)
        # print("inval_out.shape==", inval_out.shape)
        
        # print("img_out.dtype: ", img_out.dtype)
        # print("sunny_out.dtype: ", sunny_out.dtype)
        # print("inval_out.dtype: ", inval_out.dtype)
        
        loss_dict = {}
        if self.parameterization != "eps":
            target_img = x_start
        else:
            target_img = noise[:,0:1,...]
        target_img = target_img.squeeze(1).float()
        target_sunny = mask['sunny_mask'].squeeze(1).float()
        target_inval = mask['ori_invalid_mask'].squeeze(1).float()
        # print("target_img.shape==", target_img.shape)
        # print("target_sunny.shape==", target_sunny.shape)
        # print("target_inval.shape==", target_inval.shape)
        
        
        # print("target_img.dtype: ", target_img.dtype)
        # print("target_sunny.dtype: ", target_sunny.dtype)
        # print("target_inval.dtype: ", target_inval.dtype)
        
        loss = self.get_loss_masked(img_out, sunny_out, inval_out, target_img, target_sunny, target_inval, alpha_sunny, alpha_inval, mean=False)
        # print("loss.shape: ", loss.shape)
        loss = loss.mean(dim=[1,2])
        sunny_acc = ((sunny_out > 0.5).float() == target_sunny).float().mean()
        inval_acc = ((inval_out > 0.5).float() == target_inval).float().mean()
        
        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})
        
        loss_dict.update({f'{log_prefix}/sunny_acc': sunny_acc})
        loss_dict.update({f'{log_prefix}/inval_acc': inval_acc})
        
        return loss, loss_dict

        

    def forward(self, x, *args, **kwargs): 
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        if not self.use_mask:    
            assert False, "This model is not designed for non-masked training."
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
            return self.p_losses(x, t, *args, **kwargs)
        else:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
            return self.p_losses_masked(x, t, *args, **kwargs)

    def get_input(self, batch):
        x = batch["x"]
        cond = batch["cond"]
        mask = batch["mask"]
        
        # if len(x.shape) == 3:
        #     x = x[..., None]
        # x = rearrange(x, 'b h w c -> b c h w') #xsm: need to change shape?
        # x = x.to(memory_format=torch.contiguous_format).float()
        return x, cond, mask

    def shared_step(self, batch):
        if self.use_mask:
            x, cond, mask = self.get_input(batch)
            loss, loss_dict = self(x, cond=cond, mask=mask)
        return loss, loss_dict

    def training_step(self, batch):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        x, cond, mask = self.get_input(batch)
        log = dict()
        # x = batch["x"]
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        cond=cond.to(self.device)[:N]
        mask = mask["invalid_mask"].to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(cond,mask,batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=self.lr)
        return opt