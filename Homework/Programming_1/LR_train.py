import argparse
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from model import LogisticRegression_Classifier

seed = 621
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Don't change batch size
batch_size = 64

## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('./data/mnist', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
test_data = datasets.MNIST('./data/mnist', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
subset_indices = ((train_data.train_labels == 0) + (train_data.train_labels == 1)).nonzero().reshape(-1)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                           sampler=SubsetRandomSampler(subset_indices))

subset_indices = ((test_data.test_labels == 0) + (test_data.test_labels == 1)).nonzero().reshape(-1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                          sampler=SubsetRandomSampler(subset_indices))

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=30, type=int, help="Number of epochs to be trained")
parser.add_argument("--gpu", default=0, type=int, help="GPU to use")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--opt", default='SGD', choices=['SGD', 'SGDm'])
args = parser.parse_args()

num_epochs = args.epoch

# Check device status
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name())
print('Device number:', torch.cuda.device_count())
print(torch.cuda.get_device_properties(device))

# Set the GPU device
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(args.gpu)

model = LogisticRegression_Classifier(28 * 28, 2)
criterion = nn.CrossEntropyLoss()
if args.opt == 'SGD':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0)
else:
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)

save_path = f"./LR_{args.lr}_{args.opt}_{args.epoch}/"
if not os.path.exists(save_path):
    os.mkdir(save_path)

progress_bar = tqdm(range(num_epochs * len(train_loader)))

# Training the Model
# Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.
record_dict = {'trn_loss': [], 'val_loss': [], 'trn_acc': [], 'val_acc': [], 'trn_iter_loss': [], 'val_iter_loss': []}
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    trn_total_pred, trn_total_label = [], []
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        record_dict['trn_iter_loss'].append(loss.item())

        predicted_answer = torch.argmax(outputs, dim=-1)
        truth_answer = labels.detach().cpu()
        trn_total_pred.extend(predicted_answer.tolist())
        trn_total_label.extend(truth_answer.tolist())

        progress_bar.update(1)

    trn_acc = float(100 * accuracy_score(trn_total_label, trn_total_pred))
    trn_loss = total_loss / len(train_loader)
    record_dict['trn_acc'].append(trn_acc)
    record_dict['trn_loss'].append(trn_loss)

    # Test the Model
    model.eval()
    correct = 0.
    total = 0.
    eval_total_loss = 0

    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))

        outputs = model(images)
        prediction = torch.argmax(outputs, dim=-1)

        loss = criterion(outputs, labels)
        eval_total_loss += (loss.item() / len(labels))
        record_dict['val_iter_loss'].append(loss.item() / len(labels))

        correct += (prediction.view(-1).long() == labels).sum()
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

np.save('{}/record.npy'.format(save_path), record_dict)
torch.save(model.state_dict(), '{}/ckpt.pt'.format(save_path))
