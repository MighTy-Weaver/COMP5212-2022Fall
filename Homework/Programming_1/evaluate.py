import matplotlib.pyplot as plt
import numpy as np

lr_choices = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
lr_choices_str = ['5e-2', '1e-2', '5e-3', '1e-3', '5e-4', '1e-4', '5e-5', '1e-5']
opt_choices = ['SGD', 'SGDm']

# for ind, lr in enumerate(lr_choices):
#     lr_record = np.load('./LR_{}_SGD_10/record.npy'.format(lr), allow_pickle=True).item()
#     svm_record = np.load('./SVM_{}_SGD_10/record.npy'.format(lr), allow_pickle=True).item()
#     lr_record_2 = np.load('./LR_{}_SGDm_10/record.npy'.format(lr), allow_pickle=True).item()
#     svm_record_2 = np.load('./SVM_{}_SGDm_10/record.npy'.format(lr), allow_pickle=True).item()
#     num_string = "{" + lr_choices_str[ind] + "}"
#     print("\\num{} &  {}\% &  {}\% &  {}\% & {}\% \\\\".format(num_string, round(max(lr_record['val_acc']), 2),
#                                                                round(max(lr_record_2['val_acc']), 2),
#                                                                round(max(svm_record['val_acc']), 2),
#                                                                round(max(svm_record_2['val_acc']), 2)
#                                                                ))

plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = 17, 12
template = ['./LR_{}_SGD_10/record.npy', './LR_{}_SGDm_10/record.npy', './SVM_{}_SGD_10/record.npy',
            './SVM_{}_SGDm_10/record.npy']
for i in range(4):
    plt.subplot(2, 2, i + 1)
    for lr in [0.05, 0.005, 0.0005]:
        record = np.load(template[i].format(lr), allow_pickle=True).item()
        plt.plot(range(len(record['trn_loss'])), record['trn_loss'], label="LR = {}".format(lr))
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
plt.show()
