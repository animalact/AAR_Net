# -*- coding: utf-8 -*-
import os
import time

import torch
import easydict
from sklearn.model_selection import train_test_split
from lib.dataset.aar_transformer_dataset import trainDataset
from lib.models.AARNet_Transformer import TransformerModel
from lib.core.train_transformer import train, valid


if __name__ == "__main__":
    args = easydict.EasyDict({
        "batch_size": 2,
        "epoch": 100,
        "loss_interval": 500,
        "split_training": False,
        "data_path": 'data/keypoints_npy_clipped',
        "model_name": 'transformer-action-detect.pth',
        "model_path": 'weights/transformer/'})

    dataset = trainDataset(args.data_path)
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=34)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_save_path = os.path.join(args.model_path, args.model_name)


    input_size = 30
    nhead = 2
    hidden_size = 128
    n_layers = 4
    output_size = dataset.label_num
    angle_num = dataset.angle_num
    dropout = 0.1

    net = TransformerModel(input_size, nhead, hidden_size, n_layers, output_size, angle_num, dropout)
    # net.load_state_dict(torch.load(model_save_path)) 
    net.to(device)
    net = net.float()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    valid_best = dict(epoch=0, total_acc=0)

    train_losses = []
    valid_losses = []
    valid_accuracy = []

    for epoch in range(0, args.epoch):
        epoch_start_time = time.time()

        net.train()
        loss = train(train_dataloader, net, optimizer, args, epoch)
        
        net.eval()
        valid_loss, acc = valid(valid_dataloader, net, epoch, args, valid_best, output_size)
        print("end of the epoch in %f seconds" % (time.time() - epoch_start_time))

        scheduler.step()
        train_losses.append(loss)
        valid_losses.append(valid_loss)
        valid_accuracy.append(acc)

        torch.save(net.state_dict(), 'weights/transformer/transformer-action-detect.pth')

    print('[Best epoch: %d]' % valid_best['epoch'])
    print('Accuracy of %5s: %.2f %%' % ('total', valid_best['total_acc']))
    print('Done')

    net.eval()

