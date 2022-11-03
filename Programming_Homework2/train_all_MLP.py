import os

for activation in ['relu', 'sigmoid', 'elu', 'tanh', 'lrelu']:
    os.system("python train.py --gpu 6 --activation {} --model MLP".format(activation))
