import argparse
import os
import glob
import random
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
        img = Image.open(filename).convert('RGB')
        w, h = img.size
        w //= 2
        img_data = img.crop((0, 0, w, h))
        img_label_rgb = img.crop((w, 0, 2 * w, h))
        img_label_rgb_px = img_label_rgb.load()
        img_label = Image.new('L', (w, h))
        img_label_px = img_label.load()
        for x in range(w):
            for y in range(h):
                img_label_px[x, y] = CityscapesDataset.class_to_idx[img_label_rgb_px[x, y]]
        if i >= len(filenames) * training_ratio and subdir == 'Training':
            subdir = 'Test'
        img_data.save(os.path.join(dst_dir, subdir, f'{i:04}D.png'))
        img_label.save(os.path.join(dst_dir, subdir, f'{i:04}L.png'))
        print(f'{i / len(filenames) * 100:.2f}%', end='\r', flush=True)
    print('done!  ')


def main():
    parser = argparse.ArgumentParser(description='Prepare cityscapes dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', help='directory containing original dataset', required=True)
    parser.add_argument('--destination', default='data', help='directory for processed images')
    parser.add_argument('--ratio', default=0.7, help='ratio for splitting to training and validation sets')
    args = parser.parse_args()
    preprocess_data(args.destination, args.source, args.ratio)


if __name__ == "__main__":
    main()
