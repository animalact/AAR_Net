import time

import torch
from torch.utils.data import DataLoader

from lib.core.train_lstm import train
from lib.models.AARNet_LSTM import *
from lib.dataset.aar_lstm_dataset import AARDataset

if __name__ == "__main__":
    # select = [6,11,13]
    select = [2,7,12]
    train_dataset = AARDataset(data_path="./data/", category="cat", anno_path='./data/train', frame_thr=30, skip=15, select=select)
    test_dataset = AARDataset(data_path="./data/", category="cat", anno_path='./data/test', frame_thr=30, skip=1000, select=select, test=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0)

    net_cnn = CNN().cuda()
    # net_cnn = CNN3D().cuda()
    net_wellknown = Net(name="mobilenet2").cuda()
    net_lstm = LSTM(len(select)).cuda()


    train(net_wellknown, net_lstm, train_loader, test_loader)

