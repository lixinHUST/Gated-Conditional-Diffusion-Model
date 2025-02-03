
import numpy as np
import sys
import time
import torch
from contextlib import nullcontext
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from torch import autocast
from torchvision import transforms
import os

_GPU_INDEX = 0

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, cc_mask,cc_mass,additional_feature):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope(cc_mask.device):
        with model.ema_scope():
            c = model.get_learned_conditioning(cc_mass).tile(n_samples, 1, 1)
            additional_feature=additional_feature.tile(n_samples, 1, 1)
            c = model.cc_projection(c, additional_feature)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((cc_mask.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                    uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def main_run(model,sampler, cc_mask,cc_mass,additional_feature,
             scale=3.0, n_samples=10, ddim_steps=50, ddim_eta=1.0,
             precision='fp32', h=256, w=256):

    x_samples_ddim = sample_model(model, sampler, precision, h, w,
                                    ddim_steps, n_samples, scale, ddim_eta, cc_mask,cc_mass,additional_feature)

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)).convert('L'))
    return output_ims

import scipy.ndimage
def gaussian_smooth_and_normalize(hard_labels, sigma=1.0):
        """
        Apply Gaussian smoothing to a segmentation mask with labels {0, 1, 2},
        and normalize the resulting soft labels to the range [0, 1].
        
        Parameters:
        - hard_labels (np.array): The hard label segmentation map with values {0, 1, 2}.
        - sigma (float): Standard deviation of the Gaussian kernel for smoothing. 
        
        Returns:
        - soft_labels (np.array): Softened label map with smoothed boundaries, scaled to [0, 1].
        """
        # Apply Gaussian filter to the hard labels
        smoothed_labels = scipy.ndimage.gaussian_filter(hard_labels.astype(np.float32), sigma=sigma)
        
        # Normalize the smoothed values to the range [0, 1]
        soft_labels = (smoothed_labels - smoothed_labels.min()) / (smoothed_labels.max() - smoothed_labels.min())
        
        return soft_labels

def run_demo(
        device_idx=_GPU_INDEX,
        ckpt='logs/checkpoints/last.ckpt',
        config='configs/train.yaml',
        imgdir='DATA_FOLDER/images/test',
        save_dir='test_results',
        scale=3.0,
        seed=42):
    
    from pytorch_lightning import seed_everything
    seed_everything(seed)

    print('sys.argv:', sys.argv)
    if len(sys.argv) > 1:
        print('old device_idx:', device_idx)
        device_idx = int(sys.argv[1])
        print('new device_idx:', device_idx)

    device = f'cuda:{device_idx}'
    config = OmegaConf.load(config)
    
    print('Instantiating LatentDiffusion...')
    model = load_model_from_config(config, ckpt, device=device)
    sampler = DDIMSampler(model)
    names=os.listdir(imgdir)
    print('train data:',len(names))
    print('save_dir:',save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    labelfile = '/memory/a100/Datasets/SegmentationGuidedDiffusion/pair_masks/files/testset_normalized.csv'
    import pandas as pd
    label_array=np.array(pd.read_csv(labelfile))
    label_dict={}
    for i in range(len(label_array)):
        label_dict[label_array[i][0].replace('.nii.gz','.png')] = label_array[i][1:]

    mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),  
        ])
    for name in names:
        cc_img_path = os.path.join(imgdir, name)
        cc_mask_path = cc_img_path.replace('images','masks')
        cc_mask = mask_transform(Image.open(cc_mask_path).convert("L"))

        hardlabel=np.array(cc_mask)
        softlabel=gaussian_smooth_and_normalize(hardlabel, sigma=1.5)
        
        cc_mask = torch.FloatTensor(softlabel).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)
        cc_mass = torch.FloatTensor((hardlabel==2).astype(np.float32)).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)

        if name in label_dict:
            additional_feature= torch.FloatTensor((label_dict[name]).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        else:
            additional_feature= torch.FloatTensor((np.array([0]*67)).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        pred_cc=main_run(model, sampler,cc_mask,cc_mass,additional_feature,scale=scale,n_samples=1)
        pred_cc[0].save(os.path.join(save_dir, name))
 

if __name__ == '__main__':
    run_demo()
