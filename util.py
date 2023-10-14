from models.model import UNet
import torch

def get_unet_from_args(args):
    """Initializes a UNet model based on parsed arguments."""
    return UNet(ch=args.ch, 
                out_ch=args.out_ch, 
                ch_mult=args.ch_mult, 
                num_res_blocks=args.num_res_blocks, 
                attn_resolutions=args.attn_resolutions, 
                dropout=args.dropout, 
                resamp_with_conv=args.resamp_with_conv, 
                in_channels=args.in_channels, 
                resolution=args.resolution, 
                use_timestep=args.use_timestep, 
                use_linear_attn=args.use_linear_attn, 
                attn_type=args.attn_type)