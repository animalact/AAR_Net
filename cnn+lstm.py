import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import h5py
import numpy as np
import pandas as pd


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(2, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 30, 5)

    def forward(self, i):
        x = i.contiguous().view(-1, i.shape[2], i.shape[3], i.shape[4])
        x = F.relu(self.conv1(x))
        x = nn.MaxPool2d(3)(x)
        x = F.relu(self.conv2(x))
        x = nn.MaxPool2d(3)(x)
        x = F.relu(self.conv3(x))
        x = nn.AvgPool2d(3)(x)
        x = x.view(i.shape[0], i.shape[1], -1)

        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(1470, 100)
        self.fc = nn.Linear(30*100, 3)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return F.log_softmax(x)

def getH5data(h5file, vid_name):

    with h5py.File(h5file, "r") as f:  # open file
        data = f[vid_name]['data'][:]  # load all data [frame, 256, 256]
        length = data.shape[0]  # get frame length ex. 142
        start_idx = length // 2 - 30 // 2  # 71-15 = 56
        end_idx = length // 2 + 30 // 2  # 71+15 = 86
        cliped = data[start_idx:end_idx]  # clip data to [30, 256, 256]
    return cliped

class ImageFolder(Dataset):
    def __init__(self, folder_path):
        self.data = pd.read_csv(folder_path)
        self.files = self.data['filename']

    def __getitem__(self, index):
        arr = np.zeros((30, 256, 256))
        npy = np.load('./label_yerim/label_9/' + str(self.files[index]) + '.npy')
        name = "./source_9.h5"
        for i in range(30):
            for j in range(0, 15, 2):
                arr[(npy[i][j + 1] // self.data['heigth'][index]) * 255, (npy[i][j] // self.data['width'][index]) * 255] = i + 1
        rgb_batch = arr[np.newaxis,...]
        da = getH5data(name, self.files[index])[np.newaxis, ...]
        arr = np.concatenate((da, rgb_batch), axis=0)
        action_id = self.data['action_id'][index]
        action_id = action_id//3
        label = np.zeros(3)
        label[action_id] = 1
        return arr, label

    def __len__(self):
        return len(self.data)


def train(model1, model2, trainloader):
    model1.train()
    model2.train()

    optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.01)
    loss_fn = nn.NLLLoss()
    for epoch in range(500):
        for imgs,labels in trainloader:
            imgs, labels = torch.Tensor(imgs.float()).cuda(),torch.Tensor(labels.float()).cuda()
            imgs = imgs.transpose(1,2)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            features = model1(imgs)
            out = model2(features)

            loss = loss_fn(out,torch.max(labels, 1)[1].long())
            loss.backward()

            optimizer1.step()
            optimizer2.step()

            torch.save(model1.state_dict(), './save_param1.pth')
            torch.save(model2.state_dict(), './save_param2.pth')

        if epoch % 10 == 0:
            print(epoch+1, loss.cpu().detach().numpy())


if __name__ == "__main__":
    dataset = ImageFolder(folder_path='./new_csv.csv')
    train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=8)

    net_cnn = CNN().cuda()
    net_lstm = LSTM().cuda()

    train(net_cnn, net_lstm, train_loader)

