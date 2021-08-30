import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataset import random_split
import h5py
import numpy as np
import pandas as pd

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(2, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 30, 3)
        self.conv4 = nn.Conv2d(30, 40, 3)

        # self.conv1 = nn.Conv2d(2, 10, 5)
        # self.conv2 = nn.Conv2d(10, 20, 5)
        # self.conv3 = nn.Conv2d(20, 30, 5)

    def forward(self, i):
        x = i.contiguous().view(-1, i.shape[2], i.shape[3], i.shape[4])
        x = F.relu(self.conv1(x))
        x = nn.MaxPool2d(3)(x)
        x = F.relu(self.conv2(x))
        x = nn.MaxPool2d(3)(x)
        x = F.relu(self.conv3(x))
        x = nn.MaxPool2d(3)(x)
        x = F.relu(self.conv4(x))
        x = nn.AvgPool2d(3)(x)
        x = x.view(i.shape[0], i.shape[1], -1)

        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(160,100) #1470, 100)
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
        npy = np.load('./label_yerim/label_all/' + str(self.files[index]) + '.npy')
        if self.data['action_id'][index]==2:
            name = "./source_9.h5"
        elif self.data['action_id'][index]==12:
            name = "./source_7.h5"
        elif self.data['action_id'][index]==7:
            name = "./source_8.h5"

        for i in range(30):
            for j in range(0, 15, 2):
                arr[(npy[i][j + 1] // self.data['heigth'][index]) * 255, (npy[i][j] // self.data['width'][index]) * 255] = i + 1
        rgb_batch = arr[np.newaxis,...]
        da = getH5data(name, self.files[index])[np.newaxis, ...]
        arr = np.concatenate((da, rgb_batch), axis=0)
        action_id = self.data['action_id'][index]

        action_id = action_id%3 #12->0 7->1 2->2
        label = np.zeros(3)
        label[action_id] = 1
        return arr, label

    def __len__(self):
        return len(self.data)

def test(model1, model2, testloader,epoch):
    model1.load_state_dict(torch.load('./save_param1_' + str(epoch)+ '.pth'))
    model2.load_state_dict(torch.load('./save_param2_' + str(epoch)+ '.pth'))

    model1.eval()
    model2.eval()

    cnt = 0
    cnt0, cnt1, cnt2 = 0, 0, 0
    cnt0_, cnt1_, cnt2_ = 0, 0, 0
    for imgs, labels in testloader:
        imgs, labels = torch.Tensor(imgs.float()).cuda(), torch.Tensor(labels.float()).cuda()
        imgs = imgs.transpose(1, 2)

        features = model1(imgs)
        out = model2(features)

        if torch.max(labels, 1)[1] == 0:
            cnt0 += 1
            if torch.max(out, 1)[1] == 0:
                cnt0_ += 1
        elif torch.max(labels, 1)[1] == 1:
            cnt1 += 1
            if torch.max(out, 1)[1] == 1:
                cnt1_ += 1
        elif torch.max(labels, 1)[1] == 2:
            cnt2 += 1
            if torch.max(out, 1)[1] == 2:
                cnt2_ += 1
        if torch.max(out, 1)[1] == torch.max(labels, 1)[1]:
            cnt += 1

    print(cnt0_ / cnt0, cnt1_ / cnt1, cnt2_ / cnt2, cnt / 840)
def train(model1, model2, trainloader,testloader):

    optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

    loss_fn = nn.NLLLoss()

    for epoch in range(50):
        model1.train()
        model2.train()
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

        torch.save(model1.state_dict(), './save_param1_' + str(epoch)+ '.pth')
        torch.save(model2.state_dict(), './save_param2_' + str(epoch)+ '.pth')
        test(model1,model2,testloader,epoch=epoch)

        print("epoch:", epoch," _ loss:", loss.cpu().detach().numpy())


if __name__ == "__main__":
    dataset = ImageFolder(folder_path='./three.csv')
    # data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=8)
    train_set, val_set = random_split(dataset, [3400, 800])
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=True, num_workers=8)

    net_cnn = CNN().cuda()
    net_lstm = LSTM().cuda()

    train(net_cnn, net_lstm, train_loader,val_loader)

