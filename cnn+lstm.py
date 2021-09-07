import os.path
import random
import time

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

        self.conv1 = nn.Conv2d(3, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 30, 3)
        self.conv4 = nn.Conv2d(30, 40, 3)

        # self.conv1 = nn.Conv2d(2, 10, 5)
        # self.conv2 = nn.Conv2d(10, 20, 5)
        # self.conv3 = nn.Conv2d(20, 30, 5)

    def forward(self, i):
        #(batch, 3, 30, 256, 256)
        x = torch.transpose(i, 1, 2)
        x = x.contiguous().view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = F.relu(self.conv1(x))
        x = nn.MaxPool2d(3)(x)
        x = F.relu(self.conv2(x))
        x = nn.MaxPool2d(3)(x)
        x = F.relu(self.conv3(x))
        x = nn.MaxPool2d(3)(x)
        x = F.relu(self.conv4(x))
        x = nn.AvgPool2d(3)(x)
        x = x.view(i.shape[0], i.shape[2], -1)

        return x

class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 9, 2, padding=1)
        self.conv2 = nn.Conv3d(9, 27, 2, padding=1)
        self.conv3 = nn.Conv3d(27, 81, 2, padding=1)
        self.fc = nn.Linear(81*81, 30*160)  #(batch, ch, last) -> (batch, frame, outch)


    def forward(self, i):
        # x = i.contiguous().view(-1, i.shape[2], i.shape[3], i.shape[4])
        # print(x.shape)
        x = i
        x = F.relu(self.conv1(x))
        x = nn.MaxPool3d(3)(x)
        x = F.relu(self.conv2(x))
        x = nn.MaxPool3d(3)(x)
        x = F.relu(self.conv3(x))
        x = nn.AvgPool3d(3)(x)
        x = x.view(i.shape[0], -1)
        x = self.fc(x)
        x = x.view(i.shape[0], i.shape[2], -1)
        return x



class LSTM(nn.Module):
    def __init__(self, outsize):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(160, 100) #1470, 100)
        self.fc = nn.Linear(30*100, outsize)

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
    def __init__(self, data_path, anno_path, category="cat", frame_thr=None, size=256, skip=None, select=None):
        self.data_path = data_path
        self.anno_path = anno_path
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
        zero_arr = np.zeros((1, 30, 256, 256))
        arr = np.concatenate((outline_arr, keypoints_arr, zero_arr), axis=0)
        action_id = item_info["action_id"]

        label = np.zeros(len(self.select))
        label[self.labels[str(action_id)]] = 1
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
            label_nums = [2,3,4,5,6]

        for label_num in label_nums:
            label_folder = os.path.join(os.path.abspath(self.anno_path), f'label_{label_num}.csv')
            if not os.path.exists(label_folder):
                raise FileNotFoundError

            datum = pd.read_csv(label_folder)
            if self.select:
                where = datum['action_id'].isin(self.select)
                datum = datum[where]
                self.labels = {}
                for label, act in enumerate(self.select):
                    self.labels[str(act)] = label

            if data is None:
                data = datum
            else:
                data = pd.concat([data, datum], ignore_index=True)

        return data

    def _dataAugmentation(self):
        new_data_dict = {}
        if not self.frame_thr:
            return
        col = self.data.columns.to_list()
        col += ["s", "e"]

        len_new = 0
        for id, info in self.data.iterrows():
            if id % 10 == 0:
                print("", end="\r")
                print(f"{id} / {len(self.data)} -> new : {len_new} created", end="")
            clipped_num = (info.frames-self.frame_thr) // self.skip + 1
            if range == 0:
                continue
            for count in range(clipped_num):
                new_info = info.copy()
                new_info["s"] = count*self.skip
                new_info["e"] = count*self.skip + self.frame_thr
                if not new_data_dict.get(new_info['action_id'], None):
                    new_data_dict[new_info['action_id']] = []

                new_data_dict[new_info['action_id']].append(new_info)
                len_new += 1
        print()
        new_data = self._dataFlatten(new_data_dict)
        new_data = pd.DataFrame(np.array(new_data), columns=col)

        self.data = new_data
        print(len(self.data))

    def _dataFlatten(self, new_data_dict):
        new_data = []
        minimum_count = min(list(map(len, new_data_dict.values())))
        print(f"data set on minimum count : {minimum_count}")
        for action_id in new_data_dict:
            data_list = new_data_dict[action_id]
            random.shuffle(data_list)
            new_data.extend(data_list[:minimum_count])
        return new_data

    def _getKeypoints(self, item_info):
        # meta data
        keypoints_label_folder = os.path.join(self.data_path, "keypoints_npy", f'label_{item_info["label_num"]}')
        keypoints_npy_path = os.path.join(keypoints_label_folder, item_info['filename'] + ".npy")  # filepath
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
            for j in range(0, 30, 2):
                x = round((npy[s+i][j] / width) * (self.size-1))
                y = round((npy[s+i][j + 1] / height) * (self.size-1))
                arr[i, y, x] = 1+i
        return arr

    def _getH5data(self, item_info):
        label_num = item_info['label_num']
        h5file = os.path.join(self.data_path, "yolact_h5", f"source_{label_num}.h5")

        with h5py.File(h5file, "r") as f:  # open file
            data = f[item_info['filename']]['data'][:]  # load all data [frame, 256, 256]
            if self.frame_thr:
                s = int(item_info["s"])
                e = int(item_info["e"])
                data = data[s:e,:,:]
        return data




def test(model1, model2, testloader,epoch):
    model1.load_state_dict(torch.load('output/weights/save_param1_' + str(epoch)+ '.pth'))
    model2.load_state_dict(torch.load('output/weights/save_param2_' + str(epoch)+ '.pth'))

    model1.eval()
    model2.eval()

    shape = None
    correct = None
    total = None
    for imgs, labels in testloader:
        imgs, labels = torch.Tensor(imgs.float()).cuda(), torch.Tensor(labels.float()).cuda()

        features = model1(imgs)
        out = model2(features)
        if shape is None:
            shape = out.shape
            correct = torch.zeros(shape[1])
            total = torch.zeros(shape[1])

        total[torch.argmax(labels,1)] += 1
        if torch.argmax(out, 1) == torch.argmax(labels, 1):
            correct[torch.argmax(labels)] += 1


    print()
    print("-"*10)
    print(correct/total)
    print(torch.sum(correct)/torch.sum(total))


def train(model1, model2, trainloader, testloader):

    optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

    loss_fn = nn.NLLLoss()
    tot_len = int(len(trainloader.dataset)//trainloader.batch_size+1)

    prev = time.time()

    for epoch in range(50):
        print(f"epoch: {epoch} is started")
        print()
        model1.train()
        model2.train()
        cur_id = 0
        for imgs, labels in trainloader:
            cur_id+=1
            prev = time.time()
            imgs, labels = torch.Tensor(imgs.float()).cuda(),torch.Tensor(labels.float()).cuda()
            # imgs -> (None, ch, frame, w, h) -> (batch, 3, 30, 256, 256)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            features = model1(imgs)
            out = model2(features)

            loss = loss_fn(out,torch.max(labels, 1)[1].long())
            loss.backward()

            optimizer1.step()
            optimizer2.step()
            print("", end="\r")
            print(f"{cur_id} / {tot_len} batch done : {round(time.time()-prev,2)}s ", end="")

        torch.save(model1.state_dict(), 'output/weights/save_param1_' + str(epoch)+ '.pth')
        torch.save(model2.state_dict(), 'output/weights/save_param2_' + str(epoch)+ '.pth')
        test(model1,model2,testloader,epoch=epoch)

        print("epoch:", epoch," _ loss:", loss.cpu().detach().numpy())


if __name__ == "__main__":
    select = [2,7,12]
    train_dataset = DataFolder(data_path="./data/", category="cat", anno_path='./data/train', frame_thr=30, skip=15, select=select)
    test_dataset = DataFolder(data_path="./data/", category="cat", anno_path='./data/test', frame_thr=30, skip=1000, select=select)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0)
    #
    net_cnn = CNN().cuda()
    # net_cnn = CNN3D().cuda()
    net_lstm = LSTM(len(select)).cuda()
    #
    train(net_cnn, net_lstm, train_loader, test_loader)

