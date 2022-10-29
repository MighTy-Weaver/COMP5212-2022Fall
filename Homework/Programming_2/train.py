import argparse
import os
import random

import numpy as np
import torch
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import AdamW

from model import CNN_Classifier
from model import MLP_Classifier

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int, help="Index of GPU to use")
parser.add_argument("--activation", default='relu', type=str, help="Type of activation function",
                    choices=['relu', 'sigmoid', 'softmax'])
parser.add_argument("--model", type=str, choices=['MLP', 'CNN'], default='MLP', help="Model structure to use")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
parser.add_argument("--epoch", default=50, type=int, help="Number of epochs to be trained")
parser.add_argument("--seed", type=int, default=621, help="Random seed to use")
args = parser.parse_args()

seed = args.seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Print GPU info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name())
print('Device number:', torch.cuda.device_count())
print(torch.cuda.get_device_properties(device))
# Set the GPU device
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(args.gpu)


def cifar_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(32, 4),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    test = datasets.CIFAR10('./', train=False,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


batch_size = 64
test_batch_size = 64

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

if args.model == 'MLP':
    model = MLP_Classifier(activation=args.activation).to(device)
else:
    model = CNN_Classifier(activation=args.activation).to(device)

optimizer = AdamW(model.parameters(), lr=args.lr)
save_path = f"./{args.model}_{args.activation}_{args.lr}_{args.epoch}/"
if not os.path.exists(save_path):
    os.mkdir(save_path)
