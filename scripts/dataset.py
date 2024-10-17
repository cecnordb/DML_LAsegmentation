import torch
from glob import glob
from torch.utils.data import Dataset
import SimpleITK as sitk


class TrainDataset(Dataset): 


    def __init__(self, data_dir, labels_dir, patch_size, transform=None):
        self.patch_size = patch_size
        self.transform = transform
        
        self.data_paths = sorted(glob(data_dir + '/*.nii'))
        self.label_paths = sorted(glob(labels_dir + '/*.nii'))

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        image  = sitk.ReadImage(self.data_paths[idx])
        label = sitk.ReadImage(self.label_paths[idx])

        image_np = sitk.GetArrayFromImage(image).astype('float32')
        label_np = sitk.GetArrayFromImage(label).astype('int64')

        image_patch, label_patch = self.random_crop_3d(image_np, label_np, self.patch_size)

        if self.transform:
            image_patch = self.transform(image_patch)

        image_tensor = torch.tensor(image_patch, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label_patch, dtype=torch.float32)

        return image_tensor, label_tensor
    
    def random_crop_3d(self, image, label, patch_size):
        z, y, x = image.shape
        pz, py, px = patch_size
        sz = torch.randint(0, z - pz, (1,)).item()
        sy = torch.randint(0, y - py, (1,)).item()
        sx = torch.randint(0, x - px, (1,)).item()
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
        
        if self.transform:
            image_np = self.transform(image_np)

        image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)

        if label_np is not None:
            label_tensor = torch.tensor(label_np, dtype=torch.int64)
            return image_tensor, label_tensor
        else:
            return image_tensor
        

