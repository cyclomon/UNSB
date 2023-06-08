import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import torchvision.transforms as transforms


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.A_paths2 = (make_dataset(self.dir_A, opt.max_dataset_size))
        # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.B_paths2 = (make_dataset(self.dir_B, opt.max_dataset_size))
        # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        # apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying)
        # do not perform resize-crop data augmentation of CycleGAN.
        # print('current_epoch', self.current_epoch)
        self.transform = get_transform(opt, convert=False)
        if self.opt.isTrain and opt.augment:
            self.transform_aug = transforms.Compose(
                [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
                 transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        else:
            self.transform_aug = None
        self.transform_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_path2 = self.A_paths2[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
            
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_path2 = self.B_paths2[index_B]
        A_img = Image.open(A_path).convert('RGB')
        A_img2 = Image.open(A_path2).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        B_img2 = Image.open(B_path2).convert('RGB')
        A_pil = self.transform(A_img)
        A_pil2 = self.transform(A_img2)
        B_pil = self.transform(B_img)
        B_pil2 = self.transform(B_img2)
        A = self.transform_tensor(A_pil)
        A2 = self.transform_tensor(A_pil2)
        B = self.transform_tensor(B_pil)
        B2 = self.transform_tensor(B_pil2)
        if self.opt.isTrain and self.transform_aug is not None:
            A_aug = self.transform_aug(A_pil)
            A_aug2 = self.transform_aug(A_pil2)
            B_aug = self.transform_aug(B_pil)
            B_aug2 = self.transform_aug(B_pil2)
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_aug': A_aug, 'B_aug': B_aug, 'A2':A2, 'B2':B2, 'A_aug2':A_aug2, 'B_aug2':B_aug2}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A2':A2, 'B2':B2}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
