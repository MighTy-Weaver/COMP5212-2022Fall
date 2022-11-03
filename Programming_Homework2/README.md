# COMP5212 Programming Homework 2

## Code Explanation

`model.py` is for building MLP and CNN models.

`train.py` is for training both models.

The command for training MLP is:

```commandline
python train.py --model MLP \ 
                --activation relu \
                --epoch 50 
                --lr 1e-3 \
                --seed 621 \
                --gpu 0
```

The command for training CNN is:

```commandline
python train.py --model CNN \ 
                --activation relu \
                --epoch 50 
                --lr 1e-3 \
                --seed 621 \
                --gpu 0
```

Possible arguments for `--model [MLP, CNN]`.

Possible arguments for `--activation [relu, lrelu, elu, sigmoid, tanh]`.

## Reproduction

Run the `train_all_MLP.py`, `train_all_CNN.py` to train 10 groups of models automatically.

After training, run `evaluate.py` to acquire all evaluation result.