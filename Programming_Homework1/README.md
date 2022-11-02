# COMP5212 Programming Homework 1

## Code Explanation

`model.py` is for building two models.

`LR_train.py` is for training logistic regression model. Parameters can be added
by `python LR_train.py --epoch 20 --lr 5e-5 --opt SGD`

`SVM_train.py` is for training support vector machine. Parameters can be added
by `python SVM_train.py --epoch 20 --lr 3e-5 --opt SGDm`

## Reproduction

Run the `train_all_models.py` to train all 32 groups of model automatically.

After training, run `evaluate.py` to acquire all evaluation result.