import argparse
import os
import random
import warnings

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import LabelField
from tqdm import tqdm

from model import BiLSTM

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=7, type=int, help="Index of GPU to use")
parser.add_argument("--glove", default="glove.6B.50d", type=str, help="Version of Glove to use",
                    choices=['glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d', 'glove.42B.300d',
                             'glove.840B.300d'])
parser.add_argument("--dropout", default=0.05, type=float, help="Dropout Rate")
parser.add_argument("--layer", default=1, type=int, help="Number of LSTM layers", choices=[1, 2])
parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate")
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

warnings.filterwarnings('ignore')

TEXT = Field(tokenize='spacy', tokenizer_language='en_core_web_trf', include_lengths=True)
LABEL = LabelField(dtype=torch.long)
train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL, train_subtrees=True,
                                                        filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train_data, vectors=args.glove, unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
embedding = TEXT.vocab.vectors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: len(x.text),
    batch_size=batch_size, device=device)

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

word_embedding_dim = int(args.glove.split('.')[-1].split('d')[0])
model = BiLSTM(emb_dim=word_embedding_dim, num_layer=args.layer, dropout=args.dropout).to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)
criterion = CrossEntropyLoss()
save_path = f"./{args.glove}_{args.layer}"

progress_bar = tqdm(range(args.epoch * len(train_iterator)))

# Training the Model
record_dict = {'trn_loss': [], 'val_loss': [], 'tst_loss': [], 'trn_acc': [], 'val_acc': [], 'tst_acc': []}
for epoch in range(1, 1 + args.epoch):
    total_loss = 0
    model.train()
    trn_total_pred, trn_total_label = 0, 0
    for i, ((text, b), labels) in enumerate(train_iterator):
        input = torch.transpose(text, 0, 1)
        data = []
        for j in input:
            data.append(torch.stack([embedding[k] for k in j]))
        data = torch.stack(data)
        input = data.to(device)
        labels = labels.to(device)

        outputs = model(input).squeeze(1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted_answer = torch.argmax(outputs, dim=-1)
        truth_answer = labels.detach().cpu()
        trn_total_pred += (predicted_answer.view(-1).long() == labels).sum()
        trn_total_label += input.shape[0]

        progress_bar.update(1)

    trn_acc = float(100 * trn_total_pred.float() / trn_total_label)
    trn_loss = total_loss / len(train_iterator)
    record_dict['trn_acc'].append(trn_acc)
    record_dict['trn_loss'].append(trn_loss)

    # Test the Model
    model.eval()
    correct = 0.
    total = 0.
    eval_total_loss = 0

    for (text, b), labels in tqdm(valid_iterator, desc='Validating'):
        input = torch.transpose(text, 0, 1)
        data = []
        for j in input:
            data.append(torch.stack([embedding[k] for k in j]))
        data = torch.stack(data)
        input = data.to(device)
        labels = labels.to(device)

        outputs = model(input).squeeze(1)
        predicted_answer = torch.argmax(outputs, dim=-1)

        loss = criterion(outputs, labels)
        eval_total_loss += loss.item()

        correct += (predicted_answer.view(-1).long() == labels).sum()
        total += input.shape[0]
    eval_loss = eval_total_loss / len(valid_iterator)
    eval_acc = float(100 * correct.float() / total)
    record_dict['val_acc'].append(eval_acc)
    record_dict['val_loss'].append(eval_loss)

    # Test the Model
    model.eval()
    test_correct = 0.
    test_total = 0.
    test_total_loss = 0

    for (text, b), labels in tqdm(test_iterator, desc='Testing'):
        input = torch.transpose(text, 0, 1)
        data = []
        for j in input:
            data.append(torch.stack([embedding[k] for k in j]))
        data = torch.stack(data)
        input = data.to(device)
        labels = labels.to(device)

        outputs = model(input).squeeze(1)
        predicted_answer = torch.argmax(outputs, dim=-1)

        loss = criterion(outputs, labels)
        test_total_loss += loss.item()

        test_correct += (predicted_answer.view(-1).long() == labels).sum()
        test_total += input.shape[0]
    test_loss = test_total_loss / len(test_iterator)
    test_acc = float(100 * test_correct.float() / test_total)
    record_dict['tst_acc'].append(test_acc)
    record_dict['tst_loss'].append(test_loss)

    progress_bar.set_postfix({
        'trn_acc': trn_acc,
        'val_acc': eval_acc,
        'tst_acc': test_acc,
    })
    print('Accuracy of the model on train images: %f%% \t valid images: %f%%\t test images: %f%%' % (
        trn_acc, eval_acc, test_acc))
    print('MAX train acc: {}\tMAX val acc:{}\tMAX tst acc:{}'.format(max(record_dict['trn_acc']),
                                                                     max(record_dict['val_acc']),
                                                                     max(record_dict['tst_acc'])))
    np.save('{}_record.npy'.format(save_path), record_dict)
