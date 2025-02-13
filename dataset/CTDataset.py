import torch
import pickle
import numpy as np
import SimpleITK as sitk
import monai.transforms as transforms
from abc import ABC
from torch.utils.data import Dataset


def pkload(fname):
    """Load a pickle file."""
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_base_transform():
    """Define base transformations for image preprocessing."""
    base_transform = [
        transforms.AddChanneld(keys="image"),  # Add channel dimension to the image
        transforms.ScaleIntensityRanged(
            keys='image', a_min=-300.0, a_max=300.0, b_min=0.0, b_max=1.0, clip=True
        )  # Scale intensity to the range [0, 1]
    ]
    return base_transform


def get_train_transform():
    """Define additional transformations for training data augmentation."""
    train_transform = [
        transforms.RandBiasFieldd(keys="image", prob=0.3, coeff_range=(0.2, 0.3)),
        # Additional augmentations can be added here
        # e.g., transforms.GibbsNoised(keys="image", alpha=0.3),
        # transforms.RandAdjustContrastd(keys="image", prob=0.3, gamma=(1.5, 2)),
        # transforms.RandAxisFlipd(keys=["image", "mask"], prob=0.3),
        # transforms.RandAffined(keys=["image", "mask"], prob=0.2),
        # transforms.RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.3)
    ]
    return train_transform


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))  # Reorder dimensions
        label = sample['mask']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'mask': label}


class CTDataset(Dataset, ABC):
    def __init__(self, data_path, k_fold: bool, train: bool, test=None):
        super(CTDataset, self).__init__()

        self.kFold = k_fold
        self.base_transform = get_base_transform()
        self.train_transform = transforms.Compose(get_base_transform() + get_train_transform())
        self.val_transform = transforms.Compose(get_base_transform())
        self.train = train
        self.test = test
        self.cts, self.images, self.segs = [], [], []

        if isinstance(data_path, (str, list)):
            # If data_path is a string, process it as a single file path
            self.data_path = data_path
        else:
            raise ValueError("data_path must be a str or a list of str")

        if not self.kFold:
            with open(self.data_path, "r", encoding="utf-8") as f:
                info = f.readlines()
            for img_info in info:
                ct, image_path, seg_path = img_info.strip().split()
                self.cts.append(ct)
                self.images.append(image_path)
                self.segs.append(seg_path)
        else:
            for img_info in self.data_path:
                ct, image_path, seg_path = img_info.strip().split()
                self.cts.append(ct)
                self.images.append(image_path)
                self.segs.append(seg_path)

        assert len(self.images) == len(self.segs), 'The number of images and labels are not equal'
        self.dataset_size = len(self.images)

    def __getitem__(self, index):
        ct_image, brain_mask, mask_info = self.load_mri_volume(index)
        brain_mask = self.preprocess_mask_labels(brain_mask)
        modalities = torch.from_numpy(ct_image.astype(np.float32))
        segmentation_mask = torch.from_numpy(brain_mask)
        ct = self.cts[index]

        data_dict = {"image": modalities, "mask": segmentation_mask}

        if self.train:
            data_dict = self.train_transform(data_dict)
        else:
            data_dict = self.val_transform(data_dict)
        if self.test:
            return ct, data_dict, mask_info
        else:
            return ct, data_dict

    def __len__(self):
        return len(self.cts)

    def load_nifi_volume(self, file_path):
        """Load a NIfTI volume using SimpleITK."""
        image = sitk.ReadImage(file_path)
        np_image = sitk.GetArrayFromImage(image)

        direction = image.GetDirection()
        origin = image.GetOrigin()
        spacing = image.GetSpacing()

        return np_image, (direction, origin, spacing)

    def load_mri_volume(self, index):
        """Load MRI volume and corresponding ground truth mask."""
        image, _ = self.load_nifi_volume(self.images[index])
        gt_mask, mask_info = self.load_nifi_volume(self.segs[index])

        return image, gt_mask, mask_info

    def preprocess_mask_labels(self, mask: np.ndarray):
        """Preprocess mask labels into separate binary masks."""
        mask_1 = np.where(mask == 1, 1, 0)
        mask_2 = np.where(mask == 2, 1, 0)
        mask_3 = np.where(mask == 3, 1, 0)
        mask_4 = np.where(mask == 4, 1, 0)

        mask = np.stack([mask_1, mask_2, mask_3, mask_4])

        return mask


if __name__ == '__main__':
    pass
