
import importlib

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

from models.model import UNet

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