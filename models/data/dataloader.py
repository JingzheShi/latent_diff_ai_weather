import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.utils.data.dataloader as dataloader
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

def create_irregular_area(starting_point, target_size, image_size):
    # Create a tensor with a single starting point
    mask = torch.zeros(image_size, image_size)
    mask[starting_point[0], starting_point[1]] = 1.0
    
    # Apply a Gaussian filter
    filter_size = int(target_size ** 0.5) # Assuming areas are roughly square
    if filter_size % 2 == 0: filter_size += 1
    gauss_filter = torch.randn(filter_size, filter_size)
    mask = F.conv2d(mask[None, None, ...], gauss_filter[None, None, ...], padding=filter_size//2)
    
    # Threshold to create an area of the desired size
    mask = (mask > mask.flatten().topk(target_size)[0][-1]).float()
    
    return mask

def create_areas_in_tensor(batch_size, image_size, N, S):
    tensor = torch.zeros(batch_size, 1, image_size, image_size)
    
    for _ in range(N):
        # Randomly select starting points for each sample in the batch
        starting_points = torch.randint(0, image_size, (batch_size, 2))
        
        # Create a mask for each sample
        masks = [create_irregular_area(starting_point, S, image_size) for starting_point in starting_points]
        
        # Add the masks to the tensor, avoiding overlaps
        try:
            tensor += torch.stack(masks).view(batch_size, 1, image_size, image_size)
        except:
            print('image_size==',image_size)
            print(batch_size)
        tensor += torch.stack(masks).view(batch_size, 1, image_size, image_size)
    
    # Clamp values to ensure they're either 0 or 1
    tensor.clamp_(0, 1)
    
    return tensor

class NpyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.root_folder = data_path
        # check how many files are in the folder.
        self.sample_filename_list = []
        self.label_filename_list = []
        self.interpolated_filename_list = []
        for file in os.listdir(self.root_folder):
            if file.endswith("sample.npy"):
                self.sample_filename_list.append(os.path.join(self.root_folder, file))
            elif file.endswith("label.npy"):
                self.label_filename_list.append(os.path.join(self.root_folder, file))
            elif file.endswith("interpolated.npy"):
                self.interpolated_filename_list.append(os.path.join(self.root_folder, file))
        self.sample_filename_list.sort()
        self.label_filename_list.sort()
        self.interpolated_filename_list.sort()
        assert len(self.sample_filename_list) == len(self.label_filename_list)
        self.has_interpolated = len(self.interpolated_filename_list)>0
    def __len__(self):
        return len(self.sample_filename_list)
    def __getitem__(self, idx):
        if self.has_interpolated:
            return dict(
                sample=torch.from_numpy(np.load(self.sample_filename_list[idx])),
                label=torch.from_numpy(np.load(self.label_filename_list[idx])[0:1]),
                interpolated=torch.from_numpy(np.load(self.interpolated_filename_list[idx])[0:1]),
            )
        else:
            sample = torch.from_numpy(np.load(self.sample_filename_list[idx]))
            return dict(
                sample=sample,
                label=torch.from_numpy(np.load(self.label_filename_list[idx])[0:1]),
                interpolated = sample,
            )

def block_sampling(sunny_mask):
    batch_size,_,h,w = sunny_mask.shape
    block_masks = create_areas_in_tensor(batch_size, h, 20,40)
    return block_masks
def pixel_sampling(sunny_mask):
    batch_size,_,h,w = sunny_mask.shape
    pixel_masks = create_areas_in_tensor(batch_size, h, 10,10)
    return pixel_masks
def sample_indicator_mask(org_invalid_mask, sunny_mask):
    # batch_size * 1.
    indicator_mask = torch.zeros_like(sunny_mask)
    new_indicator_mask = torch.zeros_like(sunny_mask)
    # two types. block-sampling and pixel-sampling.
    pixel_number = torch.sum(torch.ones_like(sunny_mask),dim=-1).sum(dim=-1).float()
    sunny_percent = torch.sum(sunny_mask,dim=-1).sum(dim=-1).float() / pixel_number
    
    new_indicator_mask_pixel = pixel_sampling(sunny_mask)
    new_indicator_mask_block = block_sampling(sunny_mask)
    
    new_indicator_mask = 1 - (1-new_indicator_mask_pixel) * (1-new_indicator_mask_block)
    new_indicator_mask = new_indicator_mask * (1-sunny_mask)
    
    org_valid_and_cloudy = (1-sunny_mask) * (1-org_invalid_mask)
    org_valid_and_cloudy_percent = torch.sum(org_valid_and_cloudy,dim=-1).sum(dim=-1).float() / pixel_number
    new_valid_and_cloudy_percent = (1-sunny_mask)*(1-org_invalid_mask) * (1-new_indicator_mask).sum(dim=-1).sum(dim=-1).float() / pixel_number
    # batch_size * 1.
    # new_indicator_mask: batch_size * 1 * h * w.
    # now I want: new_indicator_mask[b,0,:,:] = 0 if new_valid_and_cloudy_percent[b] < 0.2 or new_valid_and_cloudy_percent[b] / org_valid_and_cloudy_percent[b] < 0.6.
    new_indicator_mask = new_indicator_mask * (new_valid_and_cloudy_percent > 0.2).float()
    new_indicator_mask = new_indicator_mask * ((new_valid_and_cloudy_percent / (org_valid_and_cloudy_percent+0.5)) > 0.6).float()
    return new_indicator_mask
    

def collate_fn(batch):
    sample=torch.stack([item['sample'] for item in batch])
    label=torch.stack([item['label'] for item in batch])
    
    # print("sample.shape: ", sample.shape)
    # print("label.shape: ", label.shape)
    
    interpolated_label = torch.stack([item['interpolated'] for item in batch])
    sunny_mask = torch.zeros_like(label).long()
    org_invalid_mask = torch.ones_like(label).long()
    
    sunny_mask[label<-998] = 1
    org_invalid_mask[label>1e-7] = 0
    org_invalid_mask[label<-1e-7] = 0
    
    indicator_mask = sample_indicator_mask(org_invalid_mask, sunny_mask).long()
    invalid_mask = 1 - (1 - org_invalid_mask) * (1 - indicator_mask)
    
    exactly_interpolated_label = interpolated_label * org_invalid_mask + label * (1-org_invalid_mask)
    partially_interpolated_label = interpolated_label * (invalid_mask) + label * (1-invalid_mask)
    competely_interpolated_label = interpolated_label
    
    mask = dict(
        sunny_mask = sunny_mask,
        invalid_mask = invalid_mask,
        indicator_mask = indicator_mask,
        ori_invalid_mask = org_invalid_mask,
    )
    
    cond = sample
    
    #Use the original label
    x = label
    #x = torch.concat((label, sunny_mask, invalid_mask), dim=1) 
    return dict(
        x = x,
        cond = cond,
        mask = mask
    )

class NpyDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, train_val_test_split=(0.8, 0.1, 0.1)):
        super(NpyDataModule, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        
        dataset = NpyDataset(self.data_path)
        num_train = int(len(dataset) * self.train_val_test_split[0] + 0.5) 
        num_val = int(len(dataset) * self.train_val_test_split[1])
        num_test = len(dataset) - num_train - num_val
        print("length of dataset: ", len(dataset))
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [num_train, num_val, num_test])
        if len(self.val_dataset) == 0:
            self.val_dataset = self.train_dataset
        if len(self.test_dataset) == 0:
            self.test_dataset = self.val_dataset
        
    def from_args(cls, args):
        return cls(data_path=args.data_path,
                   batch_size=args.batch_size,
                   train_val_test_split=args.train_val_test_split)

    def prepare_data(self):
        # Placeholder for data preprocessing function
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
    
