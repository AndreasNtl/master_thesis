from torch.utils.data import Dataset
import cv2
import numpy as np


class BrainMriDataset(Dataset):
    def __init__(self, df, img_size=128):
        self.img_size = img_size
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0])
        mask = cv2.imread(self.df.iloc[idx, 1])
        label = self.df.iloc[idx, 2]
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        image = (image - 127.5) / 127.5
        mask = (mask - 127.5) / 127.5
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        return image, mask, label
#%%
