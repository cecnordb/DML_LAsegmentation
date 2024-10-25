import torch
from glob import glob
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
from monai.transforms import ScaleIntensity

class TrainDataset(Dataset): 
    """
    require_target = True => all returned patches contain atleast some of the target
    preoad_all_images = True => all images are loaded into memory at the start
    """

    def __init__(self, data_dir, labels_dir, patch_size, require_target=0, transform=None, scale_intensity=False, preload_all_images=True, num_patches_per_image=1):
        self.patch_size = patch_size
        self.transform = transform
        self.scale_intensity = ScaleIntensity(minv=0, maxv=1) if scale_intensity else None
        self.require_target = float(require_target)
        
        self.data_paths = sorted(glob(data_dir + '/*.nii'))
        self.label_paths = sorted(glob(labels_dir + '/*.nii'))

        self.preload_all_images = preload_all_images
        if preload_all_images:
            self.data = {path_: sitk.GetArrayFromImage(sitk.ReadImage(path_)).astype('float32') for path_ in self.data_paths}
            self.labels = {path_: sitk.GetArrayFromImage(sitk.ReadImage(path_)).astype('int64') for path_ in self.label_paths}
            self.bounding_boxes = {path_: self.compute_bounding_box(self.labels[path_]) for path_ in self.label_paths}

        self.num_patches_per_image = num_patches_per_image

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        if self.preload_all_images:
            image_np = self.data[self.data_paths[idx]]
            label_np = self.labels[self.label_paths[idx]]
            bounding_box = self.bounding_boxes[self.label_paths[idx]]
        else:
            image  = sitk.ReadImage(self.data_paths[idx])
            label = sitk.ReadImage(self.label_paths[idx])

            image_np = sitk.GetArrayFromImage(image).astype('float32')
            label_np = sitk.GetArrayFromImage(label).astype('int64')
            bounding_box = None
        
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        label_tensor = torch.tensor(label_np, dtype=torch.float32)

        if self.scale_intensity is not None:
            image_tensor = self.scale_intensity(image_tensor)    
        
        patches = []
        labels = []
        for _ in range(self.num_patches_per_image):
            if torch.rand(1).item() < self.require_target:
                image_patch, label_patch = self.random_crop_3d_target(image_tensor, label_tensor, self.patch_size, bounding_box)
            else:
                image_patch, label_patch = self.random_crop_3d(image_tensor, label_tensor, self.patch_size)
            
            if self.transform:
                transform_dict = {"image": image_patch, "label": label_patch}
                transformed = self.transform(transform_dict)
                image_patch, label_patch = transformed["image"], transformed["label"]

            image_patch = image_patch.unsqueeze(0)
            patches.append(image_patch)
            labels.append(label_patch)

        return torch.stack(patches, dim=0), torch.stack(labels, dim=0)
    
    def random_crop_3d(self, image, label, patch_size):
        z, y, x = image.shape
        pz, py, px = patch_size
        sz = torch.randint(0, z - pz, (1,)).item()
        sy = torch.randint(0, y - py, (1,)).item()
        sx = torch.randint(0, x - px, (1,)).item()
        return image[sz:sz+pz, sy:sy+py, sx:sx+px], label[sz:sz+pz, sy:sy+py, sx:sx+px]
    
    def random_crop_3d_target(self, image, label, patch_size, bounding_box=None):
        z, y, x = image.shape
        pz, py, px = patch_size
        
        if bounding_box is None:
            bounding_box = self.compute_bounding_box(label)

        z_patch_min, y_patch_min, x_patch_min, z_patch_max, y_patch_max, x_patch_max = bounding_box

        sz = torch.randint(z_patch_min, z_patch_max + 1, (1,)).item()
        sy = torch.randint(y_patch_min, y_patch_max + 1, (1,)).item()
        sx = torch.randint(x_patch_min, x_patch_max + 1, (1,)).item()

        return image[sz:sz+pz, sy:sy+py, sx:sx+px], label[sz:sz+pz, sy:sy+py, sx:sx+px]

    def compute_bounding_box(self, label):
        ones_indices = np.argwhere(label)
        z_min, y_min, x_min = ones_indices.min(axis=0)
        z_max, y_max, x_max = ones_indices.max(axis=0)

        z_patch_min, y_patch_min, x_patch_min = max(0, z_min - self.patch_size[0]), max(0, y_min - self.patch_size[1]), max(0, x_min - self.patch_size[2])
        z_patch_max, y_patch_max, x_patch_max = min(label.shape[0] - self.patch_size[0], z_max), min(label.shape[1] - self.patch_size[1], y_max), min(label.shape[2] - self.patch_size[2], x_max)

        return z_patch_min, y_patch_min, x_patch_min, z_patch_max, y_patch_max, x_patch_max

        
    # TODO add intensity scaling to the constructor here

class TestDataset(Dataset): 
    """
    Test dataset loads the entire image 
    """
    def __init__(self, data_dir, labels_dir=None, transform=None, scale_intensity=False):
        self.transform = transform
        self.scale_intensity = ScaleIntensity(minv=0, maxv=1) if scale_intensity else None

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
        
        image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)

        if self.scale_intensity is not None:
            image_tensor = self.scale_intensity(image_tensor)    

        if self.transform:
            image_tensor = self.transform(image_tensor)

        if label_np is not None:
            label_tensor = torch.tensor(label_np, dtype=torch.int64)
            return image_tensor, label_tensor
        else:
            return image_tensor
        

