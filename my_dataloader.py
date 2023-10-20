from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import h5py

class CrowdDataset(Dataset):
    def __init__(self, img_root, gt_dmap_root, gt_downsample = 1):
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.gt_downsample = gt_downsample

        self.img_names = [filename for filename in os.listdir(img_root) \
                          if os.path.isfile(os.path.join(img_root, filename))]
        self.n_samples = len(self.img_names)

    def __len__(self):
        return self.n_samples
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = plt.imread(os.path.join(self.img_root, img_name))
        if len(img.shape) == 2:
            img = img[:,:,np.newaxis]
            img = np.concatenate((img, img, img), 2)
        
        gt_map = h5py.File(os.path.join(self.gt_dmap_root, img_name.replace('.JPG','.h5')), 'r')
        gt_map = np.asarray(gt_map['density_map'])
        if self.gt_downsample>1:
            ds_rows = img.shape[0]//self.gt_downsample
            ds_cols = img.shape[1]//self.gt_downsample
            img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
            img = img.transpose((2,0,1))
            gt_dmap = cv2.resize(gt_map,(ds_cols,ds_rows))
            gt_dmap = gt_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample

            img_tensor = torch.tensor(img,dtype=torch.float)
            gt_dmap_tensor = torch.tensor(gt_dmap,dtype=torch.float)
        return img_tensor, gt_dmap_tensor

if __name__ == '__main__':
    img_root = 'Dataset/image'
    gt_dmap_root = 'Dataset/groundtruth'
    dataset = CrowdDataset(img_root, gt_dmap_root, 2)
    print(dataset.img_names)
    print(os.path.join(gt_dmap_root, dataset.img_names[0].replace('.JPG','.h5')))
    for i,(img,gt_dmap) in enumerate(dataset):
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(gt_dmap)
        # plt.figure()
        # if i>5:
        #     break
        print(img.shape,gt_dmap.shape)