import argparse
import os
import glob
import random
import numpy as np
from dataset import CityscapesDataset
from PIL import Image


def preprocess_data(dst_dir, src_dir, training_ratio):
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'Test'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'Training'), exist_ok=True)
    filenames = list(glob.glob(os.path.join(src_dir, '*.png')))
    random.shuffle(filenames)
    subdir = 'Training'
    for i, filename in enumerate(filenames):
        img = np.array(Image.open(filename).convert('RGB'))
        w, h = img.size
        w /= 2
        img_data = img[:, :w, :]
        img_label_rgb = img[:, w:(w * 2), :]
        img_label = np.array([[CityscapesDataset.class_to_idx[tuple(img_label_rgb[y, x])] for x in range(w)] for y in range(h)])
        if i >= len(filenames) * training_ratio and subdir == 'Training':
            subdir = 'Test'
        Image.fromarray(img_data.astype(np.uint8)).save(os.path.join(dst_dir, subdir, f'{i:04}D.png'))
        Image.fromarray(img_label.astype(np.uint8)).save(os.path.join(dst_dir, subdir, f'{i:04}L.png'))
        print(f'[{i:04} / {len(filenames)}] {filename} -> {subdir}')


def main():
    parser = argparse.ArgumentParser(description='Prepare cityscapes dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', help='directory containing original dataset')
    parser.add_argument('--destination', default=os.path.join('data', 'cityscapes'), help='directory for processed images')
    parser.add_argument('--ratio', default=0.7, help='ratio for splitting to training and validation sets')
    args = parser.parse_args()
    preprocess_data(args.destination, args.source, args.ratio)


if __name__ == "__main__":
    main()
