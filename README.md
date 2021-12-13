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
