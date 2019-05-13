import argparse
import torch
import os
from dataset import CityscapesDataset
from torch.utils.data import DataLoader
from net import CityscapesNet


def evaluate(net, device, testloader, criterion=None):
    if criterion is None:
        net.to(device)
    n_test_batches = len(testloader)
    n_classes = len(CityscapesDataset.classes)

    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    for images, labels in testloader:
        images = images.to(device)
        images_flipped = images.flip(-1)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = net(images)
            outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, n_classes)
            outputs_flipped = net(images_flipped).flip(-1)
            outputs_flipped = outputs_flipped.permute(0, 2, 3, 1).contiguous().view(-1, n_classes)
            outputs = (outputs + outputs_flipped) / 2
            labels = labels.view(-1)
            if criterion is not None:
                test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if criterion is not None:
        test_loss /= n_test_batches
        return accuracy, test_loss
    else:
        return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate accuracy of trained model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='cityscapes.pt', help='model checkpoint to use')
    parser.add_argument('--data', default=os.path.join('data', 'cityscapes', 'Test'), help='data directory')
    parser.add_argument('--device', default='cpu', help='device to use')
    parser.add_argument('--batch', default=8, help='batch size')
    args = parser.parse_args()

    net = CityscapesNet(3, len(CityscapesDataset.classes))
    net.load_state_dict(torch.load(args.model))

    testset = CityscapesDataset(os.path.join(args.data), random_flips=False)
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=True, num_workers=4)

    accuracy = evaluate(net, args.device, testloader)
    print(f'Model accuracy: {accuracy}')


if __name__ == '__main__':
    main()
