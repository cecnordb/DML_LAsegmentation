import torch
from glob import glob
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np

class TrainDataset(Dataset): 
    """
    require_target = True => all returned patches contain atleast some of the target
    preoad_all_images = True => all images are loaded into memory at the start
    """

    def __init__(self, data_dir, labels_dir, patch_size, require_target=False,transform=None, preload_all_images=True, num_patches_per_image=1):
        self.patch_size = patch_size
        self.transform = transform
        self.require_target = require_target
        
        self.data_paths = sorted(glob(data_dir + '/*.nii'))
        self.label_paths = sorted(glob(labels_dir + '/*.nii'))

        self.preload_all_images = preload_all_images
        if preload_all_images:
            self.data = {path_: sitk.GetArrayFromImage(sitk.ReadImage(path_)).astype('float32') for path_ in self.data_paths}
            self.labels = {path_: sitk.GetArrayFromImage(sitk.ReadImage(path_)).astype('int64') for path_ in self.label_paths}

        self.num_patches_per_image = num_patches_per_image

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        if self.preload_all_images:
            image_np = self.data[self.data_paths[idx]]
            label_np = self.labels[self.label_paths[idx]]
        else:
            image  = sitk.ReadImage(self.data_paths[idx])
            label = sitk.ReadImage(self.label_paths[idx])

            image_np = sitk.GetArrayFromImage(image).astype('float32')
            label_np = sitk.GetArrayFromImage(label).astype('int64')

        patches = []
        labels = []
        for _ in range(self.num_patches_per_image):
            if self.require_target:
                image_patch, label_patch = self.random_crop_3d_target(image_np, label_np, self.patch_size)
            else:
                image_patch, label_patch = self.random_crop_3d(image_np, label_np, self.patch_size)
            
            if self.transform:
                array_data = {"image_patch": image_patch, "label_patch": label_patch}
                transformed = self.transform(array_data)
                image_patch, label_patch = transformed["image_patch"], transformed["label_patch"]
                
            image_tensor = torch.as_tensor(image_patch, dtype=torch.float32).unsqueeze(0)
            label_tensor = torch.as_tensor(label_patch, dtype=torch.float32)
            patches.append(image_tensor)
            labels.append(label_tensor)

        return torch.stack(patches, dim=0), torch.stack(labels, dim=0)
    
    def random_crop_3d(self, image, label, patch_size):
        z, y, x = image.shape
        pz, py, px = patch_size
        sz = torch.randint(0, z - pz, (1,)).item()
        sy = torch.randint(0, y - py, (1,)).item()
        sx = torch.randint(0, x - px, (1,)).item()
        return image[sz:sz+pz, sy:sy+py, sx:sx+px], label[sz:sz+pz, sy:sy+py, sx:sx+px]
    
    def random_crop_3d_target(self, image, label, patch_size):
        z, y, x = image.shape
        pz, py, px = patch_size
        
        ones_indices = np.argwhere(label)
        z_min, y_min, x_min = ones_indices.min(axis=0)
        z_max, y_max, x_max = ones_indices.max(axis=0)

        z_patch_min, y_patch_min, x_patch_min = max(0, z_min - pz), max(0, y_min - py), max(0, x_min - px)
        z_patch_max, y_patch_max, x_patch_max = min(z - pz, z_max), min(y - py, y_max), min(x - px, x_max)

        sz = torch.randint(z_patch_min, z_patch_max + 1, (1,)).item()
        sy = torch.randint(y_patch_min, y_patch_max + 1, (1,)).item()
        sx = torch.randint(x_patch_min, x_patch_max + 1, (1,)).item()

        return image[sz:sz+pz, sy:sy+py, sx:sx+px], label[sz:sz+pz, sy:sy+py, sx:sx+px]

        
class TestDataset(Dataset): 
    """
    Test dataset loads the entire image 
    """
    def __init__(self, data_dir, labels_dir=None, transform=None):
        self.transform = transform
        
        self.data_paths = sorted(glob(data_dir + '/*'))
        self.label_paths = sorted(glob(labels_dir + '/*')) if labels_dir is not None else None

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        image = sitk.ReadImage(self.data_paths[index])
        image_np = sitk.GetArrayFromImage(image).astype('float32')

        if self.label_paths is not None:
            label = sitk.ReadImage(self.label_paths[index])
            label_np = sitk.GetArrayFromImage(label).astype('int64')
        else:
            label_np = None
        
        #if self.transform:
        #    image_np = self.transform(image_np)

        image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)

        if label_np is not None:
            label_tensor = torch.tensor(label_np, dtype=torch.int64)
            return image_tensor, label_tensor
        else:
            return image_tensor