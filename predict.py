import torch
import dataset
import glob
import os
import random
import argparse
from PIL import Image
from net import CityscapesNet


def predict(net, device, images):
    net.to(device)
    net.eval()
    images = images.to(device)
    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
    return predicted


def main():
    parser = argparse.ArgumentParser(description='Perform semantic segmentation on image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', default=os.path.join('data', 'Test'),
                        help='image to process, or directory to pick random image from')
    parser.add_argument('--model', default='cityscapes.pt', help='model checkpoint to use')
    parser.add_argument('--device', default='cpu', help='device to use')
    parser.add_argument('--output', default='prediction.png', help='where to save result')
    parser.add_argument('--show', action='store_true', help='display result')
    args = parser.parse_args()

    if os.path.isdir(args.image):
        filename = random.choice(glob.glob(os.path.join(args.image, '*D.png')))
    else:
        filename = args.image

    net = CityscapesNet(3, len(dataset.CityscapesDataset.classes))
    net.load_state_dict(torch.load(args.model))

    image = Image.open(filename).convert('RGB')
    w, h = image.size

    image_t = dataset.CityscapesDataset.data_transform(image)
    label_t = predict(net, args.device, image_t.unsqueeze(0))

    label = Image.new('RGB', (w, h))
    label_px = label.load()
    for x in range(w):
        for y in range(h):
            label_px[x, y] = dataset.CityscapesDataset.classes[label_t[0, y, x]]

    result = Image.new('RGB', (w * 2, h))
    result.paste(image, (0, 0))
    result.paste(label, (w, 0))
    result.save(args.output)
    if args.show:
        result.show()


if __name__ == '__main__':
    main()
