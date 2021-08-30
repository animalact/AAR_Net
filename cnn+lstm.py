import os.path

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



class DataFolder(Dataset):
    """
    folder_path : data folder path contains "keypoints_npy, yolact_h5, label_x.csv"
    category : "cat" or "dog"
    frame_thr : how many frame do you want to get
    size : image size (default: 256)
    skip : when you use
    """
    def __init__(self, folder_path, category="cat", frame_thr=None, size=256, skip=None, select=None):
        self.folder_path = folder_path
        self.frame_thr = frame_thr
        self.skip = self.frame_thr if skip is None else skip
        self.size = size
        self.select = select
        self.data = self._getData(category)
        self._dataAugmentation()

    def __getitem__(self, index):
        item_info = self.data.loc[index]

        # keypoints data loader
        keypoints_arr = self._getKeypoints(item_info)

        keypoints_arr = keypoints_arr[np.newaxis,...]
        outline_arr = self._getH5data(item_info)[np.newaxis, ...]
        arr = np.concatenate((outline_arr, keypoints_arr), axis=0)
        action_id = item_info["action_id"]

        action_id = action_id % 3     # 12->0 7->1 2->2
        label = np.zeros(3)
        label[action_id] = 1
        print(arr.shape)
        return arr, label

    def __len__(self):
        return len(self.data)

    def _getData(self, category):
        """
            # filename, action, action_id, h, w, species, breed, emo, label_num
        """
        data = None
        if category == "cat":
            label_nums = [7,8,9]
        else:
            label_nums = [1,2,3,4,5,6]

        for label_num in label_nums:
            label_folder = os.path.join(os.path.abspath(self.folder_path), f'label_{label_num}.csv')
            if not os.path.exists(label_folder):
                raise FileNotFoundError

            datum = pd.read_csv(label_folder)
            if self.select:
                where = datum['action_id'].isin(self.select)
                data =datum[where]

            if data is None:
                data = datum
            else:
                data = pd.concat([data, datum], ignore_index=True)

        return data

    def _dataAugmentation(self):
        new_data = []
        if not self.frame_thr:
            return
        col = self.data.columns.to_list()
        col += ["s", "e"]

        for id, info in self.data.iterrows():
            if id % 500 == 0:
                print(f"{id} / {len(self.data)}")
            clipped_num = (info.frames-self.frame_thr) // self.skip + 1
            if range == 0:
                continue
            for count in range(clipped_num):
                new_info = info.copy()
                new_info["s"] = count*self.skip
                new_info["e"] = count*self.skip + self.frame_thr

                new_data.append(new_info)

        new_data = pd.DataFrame(np.array(new_data), columns=col)
        self.data = new_data


    def _getKeypoints(self, item_info):
        # meta data
        label_folder = os.path.join(self.folder_path, "keypoints_npy", f'label_{item_info["label_num"]}')
        keypoints_npy_path = os.path.join(label_folder, item_info['filename'] + ".npy")  # filepath
        height = item_info['height']
        width = item_info['width']

        # load npy data (frame, 30) to img data (frame, 256, 256)
        npy = np.load(keypoints_npy_path)

        if self.frame_thr:
            s = int(item_info["s"])
            e = int(item_info["e"])
            frame_total = e-s
        else:
            s = 0
            frame_total = npy.shape[0]


        arr = np.zeros((frame_total, self.size, self.size))
        for i in range(frame_total):
            li = []
            for j in range(0, 30, 2):
                x = round((npy[s+i][j] / width) * (self.size-1))
                y = round((npy[s+i][j + 1] / height) * (self.size-1))
                li.append([x,y])
                arr[i, y, x] = 1 #i+1
        return arr

    def _getH5data(self, item_info):
        label_num = item_info['label_num']
        h5file = os.path.join(self.folder_path, "yolact_h5", f"source_{label_num}.h5")

        with h5py.File(h5file, "r") as f:  # open file
            data = f[item_info['filename']]['data'][:]  # load all data [frame, 256, 256]
            if self.frame_thr:
                s = int(item_info["s"])
                e = int(item_info["e"])
                data = data[s:e,:,:]
        return data




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
    dataset = DataFolder(folder_path='./data', skip=10, frame_thr=10, select=[2,10,11])
    ## data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=8)
    # train_set, val_set = random_split(dataset, [3400, 800])
    # train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=8)
    # val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=True, num_workers=8)
    #
    # net_cnn = CNN().cuda()
    # net_lstm = LSTM().cuda()
    #
    # train(net_cnn, net_lstm, train_loader,val_loader)

