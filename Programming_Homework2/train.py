import argparse
import os
import random

import numpy as np
import torch
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

from model import CNN_Classifier
from model import MLP_Classifier

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int, help="Index of GPU to use")
parser.add_argument("--activation", default='relu', type=str, help="Type of activation function",
                    choices=['relu', 'sigmoid', 'elu', 'tanh', 'lrelu'])
parser.add_argument("--model", type=str, choices=['MLP', 'CNN'], default='MLP', help="Model structure to use")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
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
criterion = CrossEntropyLoss()
save_path = f"./{args.model}_{args.activation}_{args.lr}_{args.epoch}"

progress_bar = tqdm(range(args.epoch * len(train_loader)))

# Training the Model
# Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.
record_dict = {'trn_loss': [], 'val_loss': [], 'trn_acc': [], 'val_acc': [], 'trn_iter_loss': [], 'val_iter_loss': []}
for epoch in range(1, 1 + args.epoch):
    total_loss = 0
    model.train()
    trn_total_pred, trn_total_label = 0, 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        record_dict['trn_iter_loss'].append(loss.item())

        predicted_answer = torch.argmax(outputs, dim=-1)
        truth_answer = labels.detach().cpu()
        trn_total_pred += (predicted_answer.view(-1).long() == labels).sum()
        trn_total_label += images.shape[0]

        progress_bar.update(1)

    trn_acc = float(100 * trn_total_pred.float() / trn_total_label)
    trn_loss = total_loss / len(train_loader)
    record_dict['trn_acc'].append(trn_acc)
    record_dict['trn_loss'].append(trn_loss)

    # Test the Model
    model.eval()
    correct = 0.
    total = 0.
    eval_total_loss = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images).squeeze(1)
        predicted_answer = torch.argmax(outputs, dim=-1)

        loss = criterion(outputs, labels)
        eval_total_loss += loss.item()
        record_dict['val_iter_loss'].append(loss.item())

        correct += (predicted_answer.view(-1).long() == labels).sum()
        total += images.shape[0]
    eval_loss = eval_total_loss / len(test_loader)
    eval_acc = float(100 * correct.float() / total)
    record_dict['val_acc'].append(eval_acc)
    record_dict['val_loss'].append(eval_loss)
    progress_bar.set_postfix({
        'trn_acc': trn_acc,
        'val_acc': eval_acc
    })
    print('Accuracy of the model on train images: %f%% \t test images: %f%%' % (trn_acc, eval_acc))
    print('MAX train acc: {}\tMAX val acc:{}'.format(max(record_dict['trn_acc']), max(record_dict['val_acc'])))

    np.save('{}_record.npy'.format(save_path), record_dict)
