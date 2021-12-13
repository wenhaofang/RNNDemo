import os
import tqdm
import torch

def save_module(save_path, module):
    state_dict = module.state_dict()
    torch.save(state_dict,save_path)

def load_module(load_path, module):
    state_dict = torch.load(load_path)
    module.load_state_dict(state_dict)

def save_sample(folder, file_prefix, true_ids, pred_ids, true_wds, pred_wds):
    true_ids_path = os.path.join(folder, file_prefix + '_' + 'true_ids.txt')
    pred_ids_path = os.path.join(folder, file_prefix + '_' + 'pred_ids.txt')
    true_wds_path = os.path.join(folder, file_prefix + '_' + 'true_wds.txt')
    pred_wds_path = os.path.join(folder, file_prefix + '_' + 'pred_wds.txt')
    with open(true_ids_path, 'w', encoding = 'utf-8') as txt_file:
        for row in true_ids:
            for item in row:
                txt_file.write(str(int(item)))
                txt_file.write('\t')
            txt_file.write('\n')
    with open(pred_ids_path, 'w', encoding = 'utf-8') as txt_file:
        for row in pred_ids:
            for item in row:
                txt_file.write(str(int(item)))
                txt_file.write('\t')
            txt_file.write('\n')
    with open(true_wds_path, 'w', encoding = 'utf-8') as txt_file:
        for sent in true_wds:
            for word in sent:
                txt_file.write(word)
            txt_file.write('\n')
    with open(pred_wds_path, 'w', encoding = 'utf-8') as txt_file:
        for sent in pred_wds:
            for word in sent:
                txt_file.write(word)
            txt_file.write('\n')

def one_hot_embedding(x, num_classes = -1):
    num_classes = max(num_classes, int(torch.max(x)) + 1)

    source_shape = list(x.shape)
    target_shape = source_shape + [num_classes]

    inputs = torch.zeros(x.numel(), num_classes, dtype = x.dtype, device = x.device)
    result = torch.scatter(inputs, 1, x.view(-1, 1), 1).reshape(target_shape).float()

    return result

def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device = device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

def train(module, loader, criterion, optimizer, device, vocab_size, clipping_hold):
    loader.reset()
    total_loss = 0
    for mini_batch in tqdm.tqdm(loader):
        source, target = mini_batch
        source = source.to(device)
        target = target.to(device)
        inputs = one_hot_embedding(source, vocab_size)
        output = module(inputs)
        epoch_loss = criterion(
            output[:, :-1].reshape(-1, output.shape[-1]),
            target[:, 1: ].reshape(-1)
        )
        total_loss = total_loss + epoch_loss
        optimizer.zero_grad()
        epoch_loss.backward()
        grad_clipping(module.parameters(), clipping_hold, device)
        optimizer.step()
    return {
        'loss': total_loss
    }

def valid(module, loader, criterion, optimizer, device, vocab_size):
    loader.reset()
    total_loss = 0
    true_ids = []
    pred_ids = []
    true_wds = []
    pred_wds = []
    with torch.no_grad():
        for mini_batch in tqdm.tqdm(loader):
            source, target = mini_batch
            source = source.to(device)
            target = target.to(device)
            inputs = one_hot_embedding(source, vocab_size)
            output = module(inputs)
            epoch_loss = criterion(
                output[:, :-1].reshape(-1, output.shape[-1]),
                target[:, 1: ].reshape(-1)
            )
            total_loss = total_loss + epoch_loss
            true_ids.append(target)
            pred_ids.append(output)
    true_ids = torch.cat(true_ids).cpu().numpy()
    pred_ids = torch.cat(pred_ids).softmax(dim = -1).argmax(dim = 1).cpu().numpy()
    true_wds = [loader.decode_id(row) for row in true_ids]
    pred_wds = [loader.decode_id(row) for row in pred_ids]
    return {
        'loss': total_loss,
        'true_ids': true_ids,
        'pred_ids': pred_ids,
        'true_wds': true_wds,
        'pred_wds': pred_wds
    }
