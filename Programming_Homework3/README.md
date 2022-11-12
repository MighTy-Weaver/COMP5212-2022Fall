# COMP5212 Programming Homework 2

## Code Explanation

`model.py` is for building the BiLSTM model.

`train.py` is for training.

The command for training is:

```commandline
python train.py --gpu 0 \
                --glove glove.840B.300d \
                --dropout 0.05 \
                --layer 2 \
                --lr 1e-5 \
                --epoch 50
```

Possible arguments
for `--glove [glove.6B.50d, glove.6B.100d, glove.6B.200d, glove.6B.300d, glove.42B.300d, glove.840B.300d]`.

## Reproduction

Train the models with six glove embeddings and 1 or 2 LSTM layers separately following the command above. After
training, run `evaluate.py` to acquire all evaluation result.