"""Cityscapes Dataloader"""
import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
from PIL import Image, ImageOps, ImageFilter

__all__ = ['CitySegmentation']


class CitySegmentation(data.Dataset):
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 2

    def __init__(self, root='./datasets/citys', split='train', mode=None, transform=None,
                 base_size=520, crop_size=480, **kwargs):
        super(CitySegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
        self.valid_classes = [0, 1]
        self._key = np.array([0, 1])  # 0 = 背景, 1 = pantograph
        self._mapping = np.array([0, 1])  # 对应的类别索引


    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _val_sync_transform(self, img, mask):
        # 隨機水平翻轉 (保持不變)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 直接固定縮放 (不使用隨機裁剪)
        target_size = (3200, 320)  
        img = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # 隨機水平翻轉
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 直接固定縮放 (不使用隨機裁剪)
        target_size = (3200, 320)  
        img = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)
        img, mask = self._img_transform(img), self._mask_transform(mask)

        return img, mask


    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        mask_array = np.array(mask)
        # print("Unique mask values before mapping:", np.unique(mask_array))

        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    
                    maskpath = os.path.join(mask_folder, filename)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = CitySegmentation()
    img, label = dataset[0]
