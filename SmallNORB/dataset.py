"""dataset.py"""

import os
import errno
import struct
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.datasets.utils import download_url, check_integrity


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class SmallNORB(data.Dataset):
    """`smallNORB <https://cs.nyu.edu/~ylclab/data/norb-v1.0-small//>`_ Dataset.
    Args:
        root (string): Root directory of dataset where processed folder and
            and  raw folder exist.
        train (bool, optional): If True, creates dataset from the training files,
            otherwise from the test files.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If the dataset is already processed, it is not processed
            and downloaded again. If dataset is only already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        info_transform (callable, optional): A function/transform that takes in the
            info and transforms it.
        mode (string, optional): Denotes how the images in the data files are returned. Possible values:
            - all (default): both left and right are included separately.
            - stereo: left and right images are included as corresponding pairs.
            - left: only the left images are included.
            - right: only the right images are included.
    """

    dataset_root = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    data_files = {
        'train': {
            'dat': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat',
                "md5_gz": "66054832f9accfe74a0f4c36a75bc0a2",
                "md5": "8138a0902307b32dfa0025a36dfa45ec"
            },
            'info': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat',
                "md5_gz": "51dee1210a742582ff607dfd94e332e3",
                "md5": "19faee774120001fc7e17980d6960451"
            },
            'cat': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat',
                "md5_gz": "23c8b86101fbf0904a000b43d3ed2fd9",
                "md5": "fd5120d3f770ad57ebe620eb61a0b633"
            },
        },
        'test': {
            'dat': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat',
                "md5_gz": "e4ad715691ed5a3a5f138751a4ceb071",
                "md5": "e9920b7f7b2869a8f1a12e945b2c166c"
            },
            'info': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat',
                "md5_gz": "a9454f3864d7fd4bb3ea7fc3eb84924e",
                "md5": "7c5b871cc69dcadec1bf6a18141f5edc"
            },
            'cat': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat',
                "md5_gz": "5aa791cd7e6016cf957ce9bdb93b8603",
                "md5": "fd5120d3f770ad57ebe620eb61a0b633"
            },
        },
    }

    raw_folder = 'raw'
    processed_folder = 'processed'
    train_image_file = 'train_img'
    train_label_file = 'train_label'
    train_info_file = 'train_info'
    test_image_file = 'test_img'
    test_label_file = 'test_label'
    test_info_file = 'test_info'
    extension = '.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, info_transform=None, download=False,
                 mode="left"):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.info_transform = info_transform
        self.train = train  # training set or test set
        self.mode = mode

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # load test or train set
        image_file = self.train_image_file if self.train else self.test_image_file
        label_file = self.train_label_file if self.train else self.test_label_file
        info_file = self.train_info_file if self.train else self.test_info_file

        # load labels
        self.labels = self._load(label_file)

        # load info files
        self.infos = self._load(info_file)

        # load right set
        if self.mode == "left":
            self.data = self._load("{}_left".format(image_file))

        # load left set
        elif self.mode == "right":
            self.data = self._load("{}_right".format(image_file))

        elif self.mode == "all" or self.mode == "stereo":
            left_data = self._load("{}_left".format(image_file))
            right_data = self._load("{}_right".format(image_file))

            # load stereo
            if self.mode == "stereo":
                self.data = torch.stack((left_data, right_data), dim=1)

            # load all
            else:
                self.data = torch.cat((left_data, right_data), dim=0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            mode ``all'', ``left'', ``right'':
                tuple: (image, target, info)
            mode ``stereo'':
                tuple: (image left, image right, target, info)
        """
        target = self.labels[index % 24300] if self.mode is "all" else self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        info = self.infos[index % 24300] if self.mode is "all" else self.infos[index]
        if self.info_transform is not None:
            info = self.info_transform(info)

        if self.mode == "stereo":
            img_left = self._transform(self.data[index, 0])
            img_right = self._transform(self.data[index, 1])
            return img_left, img_right, target, info

        img = self._transform(self.data[index])
        return img

    def __len__(self):
        return len(self.data)

    def _transform(self, img):
        # doing this so that it is consistent with all other data sets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
            # img /= 255.
        return img

    def _load(self, file_name):
        return torch.load(os.path.join(self.root, self.processed_folder, file_name + self.extension))

    def _save(self, file, file_name):
        with open(os.path.join(self.root, self.processed_folder, file_name + self.extension), 'wb') as f:
            torch.save(file, f)

    def _check_exists(self):
        """ Check if processed files exists."""
        files = (
            "{}_left".format(self.train_image_file),
            "{}_right".format(self.train_image_file),
            "{}_left".format(self.test_image_file),
            "{}_right".format(self.test_image_file),
            self.test_label_file,
            self.train_label_file
        )
        fpaths = [os.path.exists(os.path.join(self.root, self.processed_folder, f + self.extension)) for f in files]
        return False not in fpaths

    def _flat_data_files(self):
        return [j for i in self.data_files.values() for j in list(i.values())]

    def _check_integrity(self):
        """Check if unpacked files have correct md5 sum."""
        root = self.root
        for file_dict in self._flat_data_files():
            filename = file_dict["name"]
            md5 = file_dict["md5"]
            fpath = os.path.join(root, self.raw_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        """Download the SmallNORB data if it doesn't exist in processed_folder already."""
        import gzip

        if self._check_exists():
            return

        # check if already extracted and verified
        if self._check_integrity():
            print('Files already downloaded and verified')
        else:
            # download and extract
            for file_dict in self._flat_data_files():
                url = self.dataset_root + file_dict["name"] + '.gz'
                filename = file_dict["name"]
                gz_filename = filename + '.gz'
                md5 = file_dict["md5_gz"]
                fpath = os.path.join(self.root, self.raw_folder, filename)
                gz_fpath = fpath + '.gz'

                # download if compressed file not exists and verified
                download_url(url, os.path.join(self.root, self.raw_folder), gz_filename, md5)

                print('# Extracting data {}\n'.format(filename))

                with open(fpath, 'wb') as out_f, \
                        gzip.GzipFile(gz_fpath) as zip_f:
                    out_f.write(zip_f.read())

                os.unlink(gz_fpath)

        # process and save as torch files
        print('Processing...')

        # create processed folder
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # read train files
        left_train_img, right_train_img = self._read_image_file(self.data_files["train"]["dat"]["name"])
        train_info = self._read_info_file(self.data_files["train"]["info"]["name"])
        train_label = self._read_label_file(self.data_files["train"]["cat"]["name"])

        # read test files
        left_test_img, right_test_img = self._read_image_file(self.data_files["test"]["dat"]["name"])
        test_info = self._read_info_file(self.data_files["test"]["info"]["name"])
        test_label = self._read_label_file(self.data_files["test"]["cat"]["name"])

        # save training files
        self._save(left_train_img, "{}_left".format(self.train_image_file))
        self._save(right_train_img, "{}_right".format(self.train_image_file))
        self._save(train_label, self.train_label_file)
        self._save(train_info, self.train_info_file)

        # save test files
        self._save(left_test_img, "{}_left".format(self.test_image_file))
        self._save(right_test_img, "{}_right".format(self.test_image_file))
        self._save(test_label, self.test_label_file)
        self._save(test_info, self.test_info_file)

        print('Done!')

    @staticmethod
    def _parse_header(file_pointer):
        # Read magic number and ignore
        struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        return dimensions

    def _read_image_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:
            dimensions = self._parse_header(f)
            assert dimensions == [24300, 2, 96, 96]
            num_samples, _, height, width = dimensions
            
            left_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)
            right_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)

            for i in range(num_samples):

                # left and right images stored in pairs, left first
                left_samples[i, :, :] = self._read_image(f, height, width)
                right_samples[i, :, :] = self._read_image(f, height, width)

        return torch.ByteTensor(left_samples), torch.ByteTensor(right_samples)

    @staticmethod
    def _read_image(file_pointer, height, width):
        """Read raw image data and restore shape as appropriate. """
        image = struct.unpack('<' + height * width * 'B', file_pointer.read(height * width))
        image = np.uint8(np.reshape(image, newshape=(height, width)))
        return image

    def _read_label_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:
            dimensions = self._parse_header(f)
            assert dimensions == [24300]
            num_samples = dimensions[0]

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            labels = np.zeros(shape=num_samples, dtype=np.int32)
            for i in range(num_samples):
                category, = struct.unpack('<i', f.read(4))
                labels[i] = category
            return torch.LongTensor(labels)

    def _read_info_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:

            dimensions = self._parse_header(f)
            assert dimensions == [24300, 4]
            num_samples, num_info = dimensions

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            infos = np.zeros(shape=(num_samples, num_info), dtype=np.int32)

            for r in range(num_samples):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    infos[r, c] = info

        return torch.LongTensor(infos)


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
        train_data = dset(**train_kwargs)

    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
        train_data = dset(**train_kwargs)

    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset
        train_data = dset(**train_kwargs)
    
    elif name.lower() == 'smallnorb':
        path = os.path.join(dset_dir, 'smallNORB')
        train_transform = transforms.Compose([
                          transforms.Resize((image_size, image_size)),
                          transforms.ToTensor(),])
    
        train_data = SmallNORB(path, train=True, download=False, transform=train_transform)

    else:
        raise NotImplementedError
    
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=False)

    data_loader = train_loader

    return data_loader

def build_dataloaders(batch_size, train_dataset):
  # Compute validation split
  train_size = len(train_dataset)
  indices = list(range(train_size))
  split = int(np.floor(valid_size * train_size))
  np.random.shuffle(indices)
  train_idx, valid_idx = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_idx)
  
  # Create dataloaders
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             sampler=train_sampler)
  return train_loader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    dset = CustomImageFolder('data/CelebA', transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)

    images1 = iter(loader).next()
    import ipdb; ipdb.set_trace()
    
    