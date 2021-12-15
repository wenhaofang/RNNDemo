## RNN Demo

This repository includes some demo RNN(Recurrent Neural Network) models.

Note: The project refers to [动手学深度学习](https://zh.d2l.ai/)

<br/>

Datasets:

* `dataset1`: JayChou Lyrics

Models

* `model0`: LanguageModel
    * `model1`: RNNCell
    * `model2`: LSTMCell
    * `model3`: GRUCell

### Unit Test

* for loader

```shell
# loader0: JayChou Lyrics
PYTHONPATH=. python loaders/loader0.py
```

* for module

```shell
# module1: RNNCell
PYTHONPATH=. python modules/module1.py
# module2: LSTMCell
PYTHONPATH=. python modules/module2.py
# module3: GRUCell
PYTHONPATH=. python modules/module3.py
```

```shell
# module0: LanguageModel using RNNCell
PYTHONPATH=. python modules/module0.py --rnn_type rnn
# module0: LanguageModel using LSTMCell
PYTHONPATH=. python modules/module0.py --rnn_type lstm
# module0: LanguageModel using GRUCell
PYTHONPATH=. python modules/module0.py --rnn_type gru
```

### Main Process

* for train

```shell
PYTHONPATH=. python main.py --mode train
```

* for predict

```shell
PYTHONPATH=. python main.py --mode predict
```

You can change the config either in the command line or in the file `utils/parser.py`

Here are the examples for each module:

```shell
# rnn train
PYTHONPATH=. python main.py \
    --mode train \
    --name rnn \
    --rnn_type rnn
# rnn predict
PYTHONPATH=. python main.py \
    --mode predict \
    --name rnn \
    --rnn_type rnn \
    --epoch 250 \
    --prefix 离开
```

```shell
# lstm train
PYTHONPATH=. python main.py \
    --mode train \
    --name lstm \
    --rnn_type lstm
# lstm predict
PYTHONPATH=. python main.py \
    --mode predict \
    --name lstm \
    --rnn_type lstm \
    --epoch 250 \
    --prefix 离开
```

```shell
# gru train
PYTHONPATH=. python main.py \
    --mode train \
    --name gru \
    --rnn_type gru
# gru predict
PYTHONPATH=. python main.py \
    --mode predict \
    --name gru \
    --rnn_type gru \
    --epoch 250 \
    --prefix 离开
```
