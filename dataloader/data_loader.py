import random
import numpy as np
import PIL
from PIL import Image
import skimage
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
import torchvision.transforms.functional as F


class CreateDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt

        self.img_source_paths, self.img_source_size = make_dataset(opt.img_source_file)
        self.img_target_paths, self.img_target_size = make_dataset(opt.img_target_file)

        if self.opt.isTrain:
            self.lab_source_paths, self.lab_source_size = make_dataset(opt.lab_source_file)
            # for visual results, not for training
            #self.lab_target_paths, self.lab_target_size = make_dataset(opt.lab_target_file)

        self.transform_augment = get_transform(opt, True)
        self.transform_no_augment = get_transform(opt, False)

    def __getitem__(self, item):
        # 384, 1248
        index = random.randint(0, self.img_target_size - 1)
        img_source_path = self.img_source_paths[item % self.img_source_size]
        if self.opt.dataset_mode == 'paired':
            img_target_path = self.img_target_paths[item % self.img_target_size]
        elif self.opt.dataset_mode == 'unpaired':
            img_target_path = self.img_target_paths[index]
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        img_source = Image.open(img_source_path).convert('RGB')
        img_target = Image.open(img_target_path).convert('RGB')

        # pad to (384, 1248)
        '''
        img_s = np.asarray(img_source)
        img_t = np.asarray(img_target)
        height = 384
        width  = 1248

        top_pad  = height - img_s.shape[0]
        left_pad = width - img_s.shape[1]
        print(height, width, img_s.shape, img_t.shape, top_pad, left_pad)
        img_s = np.lib.pad(img_s, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)),
                          mode='constant', constant_values=0)

        top_pad  = height - img_t.shape[0]
        left_pad = width - img_t.shape[1]
        img_t    = np.lib.pad(img_t, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)),
                          mode='constant', constant_values=0)

        img_source = PIL.Image.fromarray(numpy.uint16(img_s))
        img_target = PIL.Image.fromarray(numpy.uint16(imgt))
        '''

        img_source = img_source.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)
        img_target = img_target.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)

        if self.opt.isTrain:
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]
            lab_source = Image.open(lab_source_path)
            lab_source = lab_source.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)
            lab_source = np.array(lab_source).astype('float32') / 65536
            lab_source = np.stack((lab_source,)*3, -1)
            
            # img_source, lab_source, scale = paired_transform(self.opt, img_source, lab_source)
            img_source = self.transform_augment(img_source)
            lab_source = torch.from_numpy(lab_source.transpose(2, 0, 1)).float()

            img_target = self.transform_no_augment(img_target)

            return {'img_source': img_source, 'img_target': img_target,
                    'lab_source': lab_source, 
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path
                    }

        else:
            img_source = self.transform_augment(img_source)
            img_target = self.transform_no_augment(img_target)
            return {'img_source': img_source, 'img_target': img_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    }

    def __len__(self):
        return max(self.img_source_size, self.img_target_size)

    def name(self):
        return 'T^2Dataset'


def dataloader(opt):
    datasets = CreateDataset()
    datasets.initialize(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=opt.shuffle, num_workers=int(opt.nThreads))
    return dataset

def paired_transform(opt, image, depth):
    scale_rate = 1.0
    
    if opt.flip:
        n_flip = random.random()
        if n_flip > 0.5:
            image = F.hflip(image)
            depth = F.hflip(depth)

    if opt.rotation:
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            image = F.rotate(image, degree, Image.BICUBIC)
            depth = F.rotate(depth, degree, Image.BILINEAR)
    
    return image, depth, scale_rate


def get_transform(opt, augment):
    transforms_list = []

    if augment:
        if opt.isTrain:
            transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
    transforms_list += [
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    return transforms.Compose(transforms_list)

