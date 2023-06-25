from torch.utils.data.dataset import ConcatDataset, Dataset
import torch
import os
from typing import Union
import numpy as np
from .loader import SoundLoader


# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(
        self, root_dir: Union[str, os.PathLike], feat_len=1280, loader=SoundLoader()
    ):
        self.loader = loader
        self.feat_len = feat_len
        self.root_dir = str(root_dir)
        self.data = np.array([],dtype=np.float32).reshape(0,feat_len,80)
        self.label = np.array([],dtype=np.int8).reshape(0,1)

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.data)

    def load(self, path: Union[str, os.PathLike], class_id):
        data = torch.split(self.loader.load(path), self.feat_len)
        if data[-1].shape[0] < self.feat_len: data = data[:-1]
        if len(data) == 0: return
        self.data = np.row_stack((self.data,np.stack(data)))
        self.label = np.row_stack((self.label, np.full((len(data),1),class_id)))
        return data
    
    def add(self,path,class_id):
        for file in os.listdir(path):
            self.load(os.path.join(path,file),class_id)

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
