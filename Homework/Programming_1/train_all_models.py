import os

lr_choices = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
opt_choices = ['SGD', 'SGDm']

for lr in lr_choices:
    for opt in opt_choices:
        os.system("python LR_train.py --opt {} --lr {} --epoch 10".format(opt, lr))
        os.system("python SVM_train.py --opt {} --lr {} --epoch 10".format(opt, lr))
os.system("python evaluate.py")
