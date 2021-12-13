import re
import random

import torch

class DataLoader():
    def __init__(self, file_path, min_freq, max_numb, batch_size, max_seq_len):
        self.SOS_TOKEN = '[SOS]'
        self.EOS_TOKEN = '[EOS]'
        self.UNK_TOKEN = '[UNK]'
        self.PAD_TOKEN = '[PAD]'

        self.text  = self.read_file  (file_path)
        self.vocab = self.build_vocab(self.text, min_freq, max_numb)
        self.ids   = self.encode_text(self.text)

        self.batch_size  = batch_size
        self.max_seq_len = max_seq_len

    def read_file(self, file_path):
        text = ''
        with open(file_path, 'r', encoding = 'utf-8') as txt_file:
            text = txt_file.read()
            text = re.sub(r'[^\u4e00-\u9fa5\s]', '', text) # only process Chinese
            text = re.sub(r'\s+', ' ', text)
        return text

    def build_vocab(self, text, min_freq, max_numb):
        counter = {}
        for char in text:
            counter[char] = counter.get(char, 0) + 1
        corpus = sorted(counter.items(), key = lambda item: item[1], reverse = True)
        corpus = [char for idx, (char, count) in enumerate(corpus) if count >= min_freq and idx < max_numb - 4]
        corpus.append(self.PAD_TOKEN)
        corpus.append(self.UNK_TOKEN)
        corpus.append(self.EOS_TOKEN)
        corpus.append(self.SOS_TOKEN)
        vocab = {}
        vocab['id2char'] = {idx: char for idx, char in enumerate(corpus)}
        vocab['char2id'] = {char: idx for idx, char in enumerate(corpus)}
        return vocab

    def encode_text(self, chars):
        ids = [self.vocab['char2id'].get(char, self.UNK_TOKEN) for char in chars]
        return ids

    def decode_id(self, ids):
        chars =  [self.vocab['id2char'].get(_id) for _id in ids]
        return chars

    def get_vocab_size(self):
        assert len(self.vocab['id2char']) == len(self.vocab['char2id'])
        return len(self.vocab['id2char'])

    def random_sampling(self):
        num_example = len(self.ids) // self.max_seq_len
        num_batch = num_example // self.batch_size

        example_indices = list(range(num_example))
        random.shuffle(example_indices)

        Xs = []
        Ys = []
        for i in range(num_batch):
            batch_indices = example_indices[i * self.batch_size: i * self.batch_size + self.batch_size]
            X = [self.ids[j * self.max_seq_len: j * self.max_seq_len + self.max_seq_len] for j in batch_indices]
            Y = [self.ids[j * self.max_seq_len + 1: j * self.max_seq_len + 1 + self.max_seq_len] for j in batch_indices]
            Xs.append(torch.tensor(X, dtype = torch.long))
            Ys.append(torch.tensor(Y, dtype = torch.long))

        return list(zip(Xs, Ys))

    def reset(self):
        self.curr_id = -1
        self.datas = self.random_sampling()

    def __iter__(self):
        return self

    def __next__(self):
        if  self.curr_id < len(self.datas) - 1:
            self.curr_id += 1
            return self.datas[self.curr_id]
        else:
            raise StopIteration

def get_loader(option):
    return DataLoader(
        option.data_path,
        option.min_freq,
        option.max_numb,
        option.batch_size,
        option.max_seq_len
    )

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    loader = get_loader(option)

    loader.reset()
    for mini_batch in loader:
        X, Y = mini_batch
        print(X.shape) # (batch_size, max_seq_len)
        print(Y.shape) # (batch_size, max_seq_len)
        break
