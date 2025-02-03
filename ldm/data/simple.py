from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler

# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)

def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [NfpDataset(x, image_transforms=copy.copy(tforms), default_caption="A view from a train window") for x in dirs]
    return torch.utils.data.ConcatDataset(datasets)


def make_tranforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [FolderData(p, caption_file=c, **kwargs) for (p, c) in zip(paths, caption_files)]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)



class NfpDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1


    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index+1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset=CustomDataset(train_val_test='train')
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset=CustomDataset(train_val_test='val')
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(CustomDataset(train_val_test='test'),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
import pandas as pd
import scipy.ndimage
class CustomDataset(Dataset):
    def __init__(self,img_dir='DATA_FOLDER',train_val_test='train'):
        train_val_test_dict = {'train': 'trainset_normalized.csv', 'val': 'valset_normalized.csv', 'test': 'testset_normalized.csv'}
        self.labelfile = os.path.join('files',train_val_test_dict[train_val_test])
        self.label_array=np.array(pd.read_csv(self.labelfile))
        self.label_dict={}
        for i in range(len(self.label_array)):
            self.label_dict[self.label_array[i][0].replace('.nii.gz','.png')] = self.label_array[i][1:]
       
        self.cc_img_root = os.path.join(img_dir,'images',train_val_test)
        self.images=os.listdir(self.cc_img_root)
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 保持 128x128 大小
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),  
        ])
        self.type=type
        
    def __len__(self):
        return len(self.images)
    
    def get_random_box_mask(self, image):
        y, x = image.shape[-2:]
        mask = torch.zeros((1, y, x))

        top = random.randint(0, y)
        left = random.randint(0, x)
        height = random.randint(0, y - top)
        width = random.randint(0, x - left)

        mask[:, top:top+height, left:left+width] = 1

        return 1-mask
    
    def gaussian_smooth_and_normalize(self, hard_labels, sigma=1.0):
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
    
    def __getitem__(self, idx):
        cc_img_path = os.path.join(self.cc_img_root, self.images[idx])
        cc_mask_path = cc_img_path.replace('images','masks')
        cc_image= self.image_transform(Image.open(cc_img_path).convert("RGB"))
        cc_mask = self.mask_transform(Image.open(cc_mask_path).convert("L"))
        hardlabel=np.array(cc_mask)
        softlabel=self.gaussian_smooth_and_normalize(hardlabel, sigma=1)
        
        data = {}
        data["image_target"] = cc_image
        data["image_cond"] = torch.FloatTensor(softlabel).unsqueeze(0).repeat(3,1,1)
        data["mass_cond"]= torch.FloatTensor((hardlabel==2).astype(np.float32)).unsqueeze(0).repeat(3,1,1)
        if self.images[idx] in self.label_dict:
            data["additional_feature"]= torch.FloatTensor((self.label_dict[self.images[idx]]).astype(np.float32)).unsqueeze(0)
        else:
            data["additional_feature"]= torch.FloatTensor((np.array([0]*67)).astype(np.float32)).unsqueeze(0)
        return data


class FolderData(Dataset):
    def __init__(self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(sorted(list(self.root_dir.rglob(f"*.{e}"))))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions.get(chosen, None)
            if caption is None:
                caption = self.default_caption
            filename = self.root_dir/chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        data["image"] = im

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
import random

class TransformDataset():
    def __init__(self, ds, extra_label="sksbspic"):
        self.ds = ds
        self.extra_label = extra_label
        self.transforms = {
            "align": transforms.Resize(768),
            "centerzoom": transforms.CenterCrop(768),
            "randzoom": transforms.RandomCrop(768),
        }


    def __getitem__(self, index):
        data = self.ds[index]

        im = data['image']
        im = im.permute(2,0,1)
        # In case data is smaller than expected
        im = transforms.Resize(1024)(im)

        tform_name = random.choice(list(self.transforms.keys()))
        im = self.transforms[tform_name](im)

        im = im.permute(1,2,0)

        data['image'] = im
        data['txt'] = data['txt'] + f" {self.extra_label} {tform_name}"

        return data

    def __len__(self):
        return len(self.ds)

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    tform = make_tranforms(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]



import random
import json
class IdRetreivalDataset(FolderData):
    def __init__(self, ret_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(ret_file, "rt") as f:
            self.ret = json.load(f)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.paths[index].name
        matches = self.ret[key]
        if len(matches) > 0:
            retreived = random.choice(matches)
        else:
            retreived = key
        filename = self.root_dir/retreived
        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        # data["match"] = im
        data["match"] = torch.cat((data["image"], im), dim=-1)
        return data
