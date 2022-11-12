import numpy as np
from matplotlib import pyplot as plt

glove_choices = ['glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d', 'glove.42B.300d',
                 'glove.840B.300d']
glove_str_dict = {'glove.6B.50d': 'GLOVE-6B-50dim', 'glove.6B.100d': 'GLOVE-6B-100dim',
                  'glove.6B.200d': 'GLOVE-6B-200dim', 'glove.6B.300d': 'GLOVE-6B-300dim',
                  'glove.42B.300d': 'GLOVE-42B-300dim', 'glove.840B.300d': 'GLOVE-840B-300dim'}

LSTM_layer_choices = [1, 2]
LSTM_layer_str_dict = {1: 'Single Layer', 2: 'Double Layers'}

plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = 25, 15

for l in LSTM_layer_choices:
    plt.subplot(2, 3, 3 * l - 2)
    for g in glove_choices:
        try:
            record = np.load('./{}_{}_record.npy'.format(g, l), allow_pickle=True).item()
            plt.plot(range(len(record['trn_loss'])), record['trn_loss'], label=glove_str_dict[g])
        except FileNotFoundError:
            continue
    plt.xlabel("Epoch Number")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Training Loss Curve for {} LSTM".format(LSTM_layer_str_dict[l]))

    plt.subplot(2, 3, 3 * l - 1)
    for g in glove_choices:
        try:
            record = np.load('./{}_{}_record.npy'.format(g, l), allow_pickle=True).item()
            plt.plot(range(len(record['val_acc'])), record['val_acc'], label=glove_str_dict[g])
        except FileNotFoundError:
            continue
    plt.xlabel("Epoch Number")
    plt.ylabel("Validation Set Accuracy")
    plt.legend(loc=4)
    plt.title("Validation Accuracy Curve for {} LSTM".format(LSTM_layer_str_dict[l]))

    plt.subplot(2, 3, 3 * l)
    for g in glove_choices:
        try:
            record = np.load('./{}_{}_record.npy'.format(g, l), allow_pickle=True).item()
            plt.plot(range(len(record['tst_acc'])), record['tst_acc'], label=glove_str_dict[g])
        except FileNotFoundError:
            continue
    plt.xlabel("Epoch Number")
    plt.ylabel("Testing Set Accuracy")
    plt.legend(loc=4)
    plt.title("Testing Accuracy Curve for {} LSTM".format(LSTM_layer_str_dict[l]))

try:
    plt.savefig('./loss.pdf', dpi=800, bbox_inches='tight')
    plt.clf()
except PermissionError:
    print("Please close pdf first")

glove_str_dict_table = {'glove.6B.50d': '6B-50dim', 'glove.6B.100d': '6B-100dim',
                        'glove.6B.200d': '6B-200dim', 'glove.6B.300d': '6B-300dim',
                        'glove.42B.300d': '42B-300dim', 'glove.840B.300d': '840B-300dim'}

for g in glove_choices:
    record1 = np.load('./{}_1_record.npy'.format(g), allow_pickle=True).item()
    record2 = np.load('./{}_2_record.npy'.format(g), allow_pickle=True).item()
    print(
        "{} & {} & {} & {} & {} \\\\".format(glove_str_dict_table[g], round(max(record1['val_acc']), 2),
                                             round(max(record1['tst_acc']), 2), round(max(record2['val_acc']), 2),
                                             round(max(record2['tst_acc']), 2)))
