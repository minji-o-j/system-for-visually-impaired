import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        
    def load_image(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
            
        return img_path, img
    
    def load_targets(self, index, img):
        img = torch.from_numpy(img)
        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
#         img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        
        print("pth:"+label_path)
        targets = None
        
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        return targets

    def __getitem__(self, index):
        # img_path, img, targets = self.load_mosaic(index)

        img_path, img = self.load_image(index)
        targets = self.load_targets(index, img.numpy())
        img = resize(img, self.img_size)
        
        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                '''
                print("**")
                print(self)
                print(img)
                print(index)
                print(targets)
                print("**")
                '''
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


    # def load_mosaic(self, index):
    #     # loads images in a mosaic
    #     img_path = self.img_files[index % len(self.img_files)].rstrip()

    #     labels4 = []
    #     s = self.img_size
    #     xc, yc = [int(random.uniform(s * 0.1, s * 0.9)) for _ in range(2)]  # mosaic center x, y
    #     indices = [index] + [random.randint(0, len(self.label_files) - 1) for _ in range(3)]  # 3 additional image indices
        
    #     img4 = torch.full((s, s, 3), 0)
    #     for i, index in enumerate(indices):
    #         # Load image
    #         img_path = self.img_files[index % len(self.img_files)].rstrip()

    #         # Extract image as PyTorch tensor
    #         img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    #         img = resize(img, self.img_size)
    #         img = img.permute(1, 2, 0)
            
    #         if i==0:
    #             xmin = 0; xmax = xc; ymin=yc; ymax=s
    #         elif i==1:
    #             xmin = xc; xmax = s; ymin=yc; ymax=s
    #         elif i==2:
    #             xmin = 0; xmax = xc; ymin=0; ymax=yc
    #         else:
    #             xmin = xc; xmax = s; ymin=0; ymax=yc
                
    #         img4[xmin:xmax, ymin:ymax] = img[xmin:xmax, ymin:ymax]

    #         targets = self.load_targets(index, img.numpy())
    #         print(targets)
    #         for _, i, x, y, w, h  in targets:
    #             x *= 416; y *= 416; w *= 416; h *= 416
    #             x1 = x-w/2; x2 = x+w/2
    #             x1 = np.clip(x1, xmin, xmax); x2 = np.clip(x2, xmin, xmax)
    #             y1 = y-h/2; y2 = y+h/2
    #             y1 = np.clip(y1, ymin, ymax); y2 = np.clip(y2, ymin, ymax)
    #             if x2-x1==0 or y2-y1==0:
    #                 pass
    #             else:
    #                 x = ((x1+x2)/2) / w
    #                 y = ((y1+y2)/2) / h
    #                 w = abs(x2-x1)/w
    #                 h = abs(y2-y1)/h
    #                 labels4.append(np.array([0, i, x, y, w, h]))

    #     img4 = img4.permute(2, 0, 1)
    #     labels4 = torch.from_numpy(np.array(labels4))
    #     return img_path, img4, labels4

