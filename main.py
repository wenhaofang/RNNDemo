import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

seed = 77

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils.parser import get_parser
from utils.logger import get_logger

parser = get_parser()
option = parser.parse_args()

root_path = 'result'

logs_folder = os.path.join(root_path, 'logs', option.name)
save_folder = os.path.join(root_path, 'save', option.name)
sample_folder = os.path.join(root_path, 'sample', option.name)

subprocess.run('mkdir -p %s' % logs_folder, shell = True)
subprocess.run('mkdir -p %s' % save_folder, shell = True)
subprocess.run('mkdir -p %s' % sample_folder, shell = True)

logger = get_logger(option.name, os.path.join(logs_folder, 'main.log '))

from loaders.loader0 import get_loader
from modules.module0 import get_module

from utils.misc import train, valid, predict, save_module, load_module, save_sample

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

loader = get_loader(option)

logger.info('prepare module')

module = get_module(option, loader.get_vocab_size())

logger.info('prepare envs')

vocab_size = loader.get_vocab_size()

optimizer = optim.SGD(module.parameters(), lr = option.learning_rate)
criterion = nn.CrossEntropyLoss()

if option.mode == 'train':

    logger.info('start train')

    for epoch in range(option.num_epochs):
        train_info = train(module, loader, criterion, optimizer, device, vocab_size, option.clipping_hold)
        valid_info = valid(module, loader, criterion, optimizer, device, vocab_size)
        logger.info(
            'epoch: %d, train_loss: %f, valid_loss: %f' %
            (epoch, train_info['loss'], valid_info['loss'])
        )
        if epoch % 10 == 0:
            prefix = 'epoch' + str(epoch)
            save_module(
                os.path.join(save_folder, prefix + '.ckpt'),
                module
            )
            save_sample(
                sample_folder,
                prefix,
                valid_info['true_ids'],
                valid_info['pred_ids'],
                valid_info['true_wds'],
                valid_info['pred_wds']
            )

if option.mode == 'predict':

    logger.info('start predict')

    logger.info('epoch: %d' % option.epoch)
    load_module(
        os.path.join(save_folder, 'epoch' + str(option.epoch) + '.ckpt'),
        module
    )

    logger.info('prefix: %s, seq_len: %d' % (option.prefix, option.max_seq_len))

    result = predict(module, loader, option.prefix, option.max_seq_len)

    logger.info('result: %s' % result)
