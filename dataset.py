import os
import glob
import random
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image


class CityscapesDataset(data.Dataset):
    classes = [
        (116, 17, 36), (152, 43, 150), (106, 141, 34), (69, 69, 69), (2, 1, 3), (127, 63, 126), (222, 52, 211),
        (2, 1, 140), (93, 117, 119), (180, 228, 182), (213, 202, 43), (79, 2, 80), (188, 151, 155), (9, 5, 91),
        (106, 75, 13), (215, 20, 53), (110, 134, 62), (8, 68, 98), (244, 171, 170), (171, 43, 74), (104, 96, 155),
        (72, 130, 177), (242, 35, 231), (147, 149, 149), (35, 25, 34), (155, 247, 151), (85, 68, 99), (71, 81, 43),
        (195, 64, 182), (146, 133, 92),
    ]

    class_to_idx = dict((c, i) for i, c in enumerate(classes))

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self, root, random_flips=False):
        self.data_filenames = list(glob.glob(os.path.join(root, '*D.png')))
        self.label_filenames = list(glob.glob(os.path.join(root, '*L.png')))
        self.random_flips = random_flips

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, i):
        data_img = Image.open(self.data_filenames[i]).convert('RGB')
        label_img = Image.open(self.label_filenames[i]).convert('L')
        hflip = self.random_flips and bool(random.randint(0, 1))

        data_img_t = self.data_transform(data_img)
        label_img_t = torch.from_numpy(np.array(label_img)).long()
        if hflip:
            data_img_t = data_img_t.flip(-1)
            label_img_t = label_img_t.flip(-1)

        return data_img_t, label_img_t
