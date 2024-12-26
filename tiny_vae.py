from typing import Tuple

import PIL
import numpy as np
import torch
from PIL import Image
from torch import Tensor

import folder_paths
from comfy import model_management
from comfy.taesd.taesd import TAESD

#credit for Decode goes to https://github.com/M1kep/ComfyUI-OtherVAEs, this is based on that implementation, but fixes some issues and includes the encoder
class TAESDVaeDecode:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": (folder_paths.get_filename_list("vae_approx"), {})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    OUTPUT_IS_LIST = (False,)
    CATEGORY = "conditioning"

    def __init__(self):
        self.taesd = None

    def decode(self, latent: torch.Tensor, vae: str) -> Tuple[torch.Tensor]:
        if self.taesd is None:
            self.taesd = TAESD(None, folder_paths.get_full_path("vae_approx", vae)).to(model_management.get_torch_device())

        x_sample = self.taesd.taesd_decoder((latent['samples'].to(model_management.get_torch_device()) * 0.18215))[0].detach()
        x_sample = x_sample.sub(0.5).mul(2)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        # x_sample = x_sample.movedim(1, -1)
        x_sample = x_sample.permute(1, 2, 0)

        # x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
        # x_sample = x_sample.astype(np.uint8)

        return (torch.unsqueeze(x_sample, 0),)

class TAESDVaeEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI image input (NHWC format)
                "vae": (folder_paths.get_filename_list("vae_approx"), {})
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    OUTPUT_IS_LIST = (False,)
    CATEGORY = "conditioning"

    def __init__(self):
        self.taesd = None

    def encode(self, image: torch.Tensor, vae: str) -> Tuple[dict]:
        if self.taesd is None:
            # Initialize with encoder path only since we only need encoding
            self.taesd = TAESD(
                encoder_path=folder_paths.get_full_path("vae_approx", vae),
                decoder_path=None
            ).to(model_management.get_torch_device())
        
        # Move input to the same device as the model
        device = model_management.get_torch_device()
        x_sample = image.to(device)
        
        # Convert NHWC to NCHW format for the encoder
        x_sample = x_sample.permute(0, 3, 1, 2)
        
        # Scale input from [0,1] to [-1,1]
        x_sample = (x_sample * 2.0) - 1.0
        
        # Encode the image using TAESD's encode method
        latent = self.taesd.encode(x_sample)
        
        return ({"samples": latent},)
