import matplotlib.pyplot as plt
import numpy as np

model_choices = ['MLP', 'CNN']
activation_choices = ['relu', 'lrelu', 'elu', 'sigmoid', 'tanh']
activation_str_dict = {'relu': 'ReLU', 'lrelu': 'Leaky ReLU', 'elu': 'ELU', 'sigmoid': 'Sigmoid', 'tanh': 'Tanh'}

for a in activation_choices:
    MLP_record = np.load(f'./MLP_{a}_0.001_50_record.npy', allow_pickle=True).item()
    CNN_record = np.load(f'./CNN_{a}_0.001_50_record.npy', allow_pickle=True).item()
    print("{} & {} & {} \\\\".format(activation_str_dict[a], max(MLP_record['val_acc']), max(CNN_record['val_acc'])))

plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = 20, 14
for i in range(4):
    plt.subplot(2, 2, i + 1)

plt.subplot(2, 2, 1)
for a in activation_choices:
    MLP_record = np.load(f'./MLP_{a}_0.001_50_record.npy', allow_pickle=True).item()
    plt.plot(range(len(MLP_record['trn_loss'])), MLP_record['trn_loss'], label=activation_str_dict[a])
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve for Multi Layer Perceptron")

plt.subplot(2, 2, 2)
for a in activation_choices:
    CNN_record = np.load(f'./CNN_{a}_0.001_50_record.npy', allow_pickle=True).item()
    plt.plot(range(len(CNN_record['trn_loss'])), CNN_record['trn_loss'], label=activation_str_dict[a])
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve for Convolutional Neural Network")

plt.subplot(2, 2, 3)
for a in activation_choices:
    MLP_record = np.load(f'./MLP_{a}_0.001_50_record.npy', allow_pickle=True).item()
    plt.plot(range(len(MLP_record['val_acc'])), MLP_record['val_acc'], label=activation_str_dict[a])
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Testing Accuracy")
plt.title("Testing Accuracy Curve for Multi Layer Perceptron")

plt.subplot(2, 2, 4)
for a in activation_choices:
    CNN_record = np.load(f'./CNN_{a}_0.001_50_record.npy', allow_pickle=True).item()
    plt.plot(range(len(CNN_record['val_acc'])), CNN_record['val_acc'], label=activation_str_dict[a])
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Testing Accuracy")
plt.title("Testing Accuracy Curve for Convolutional Neural Network")

plt.savefig('./loss.png', dpi=400, bbox_inches='tight')
plt.clf()
