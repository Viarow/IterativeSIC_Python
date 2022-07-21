import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
from dataset import QAM_SingleDataset
from network import InitNet


def train_single_SNR(params, SNRdB, model, loss_fn, optimizer, cuda=False):

    training_data = QAM_SingleDataset(params, SNRdB)
    amount = len(training_data)
    dataloader = DataLoader(training_data, batch_size = params['batch_size'])

    model.train()
    
    for t in range(0, params['epochs']):
        print(f"Epoch {t+1}\n------------------")
        for batch, sample in enumerate(dataloader):
            y_H = sample['input']
            x_cat = sample['label']
            if cuda:
                y_H = y_H.cuda()
                x_cat = x_cat.cuda()

            xhat_cat = model(y_H)
            loss = loss_fn(xhat_cat, x_cat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                loss, current = loss.item(), batch * params['batch_size']
                print(f"loss: {loss:>7f}  [{current:>5d}/{amount:>5d}]")


def main_loop(params, SNRdB_range, ckpt_path, cuda=False):
    model = InitNet(params['m'], params['n'])
    if cuda:
        model = model.cuda()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    for i, SNRdB in enumerate(SNRdB_range):
        print(f"SNRdB {SNRdB}\n--------------------------")
        train_single_SNR(params, SNRdB, model, loss_fn, optimizer, cuda)

    torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    params = {
        'm' : 2,
        'n' : 2,
        'mod': 4,
        'H_mean': 0,
        'H_var': 1,
        'size': 100000,
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 1e-3
    }
    SNRdB_range = np.arange(0, 35, 5)
    ckpt_path = './workspaces/Tx2Rx2_CPU.pt'
    main_loop(params, SNRdB_range, ckpt_path, cuda=False)
