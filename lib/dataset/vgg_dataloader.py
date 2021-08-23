import os

import torch
import torchvision.transforms as tr # 이미지 전처리 기능들을 제공하는 라이브러리
from torch.utils.data import DataLoader, Dataset # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
import numpy as np # 넘파이 기본 라이브러리

import h5py

class MyDataset(Dataset):
    
    def __init__(self, x_data, y_data, transform=None):
        
        self.x_data = x_data # 넘파이 배열이 들어온다.
        self.y_data = y_data # 넘파이 배열이 들어온다.
        self.transform = transform
        self.len = len(y_data)
    
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        
        if self.transform:
            sample = self.transform(sample) #self.transform이 None이 아니라면 전처리를 작업한다.
        
        return sample # 3.3과 다르게 넘파이 배열로 출력 되는 것에 유의 하도록 한다.
    
    def __len__(self):
        return self.len       


class ToTensor:
    def __call__(self, sample):
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)      # 텐서로 변환
        # inputs = inputs.permute(2,0,1)          # 크기 변환
        return inputs, torch.LongTensor(labels) # 텐서로 변환



trans = tr.Compose([ToTensor()]) # 텐서 변환 후 선형식 2x+5 연산

import json
import random

data_file= "/home/butlely/Desktop/Dataset/aihub/yolact_7/yolact_7.h5"
label_file = "/home/butlely/Desktop/Dataset/aihub/yolact_label/actionLabel.json"

frame = 20

train_images = []
train_labels = []
test_images = []
test_labels = []

def splitTrainValid(label_file):
    """
    :param label_file: {"vid_name": 0, ...}
    :return: [("name", 0), ...], [("name", 0), ...],
    """
    with open(label_file, "r") as f:
        label_data = json.load(f)
    li1 = list(label_data.items())[:2000]
    li2 = list(label_data.items())[-1776:]
    data = li1+li2
    random.shuffle(data)
    train = data[:3000]
    test = data[3000:]

    return train, test

def dataloader(label_file):
    frame = 20

    f = h5py.File(data_file, "r")
    train_labels, test_labels = splitTrainValid(label_file)
    nodata = []

    tr_x = None
    te_x = None
    print("train dataset start")
    for train_label in train_labels:
        id, label = train_label[0], train_label[1]
        vid = f.get(id, None)
        if vid is None:
            nodata.append(id)
            continue
        data = vid['data'][:]
        if data.shape[0] < frame:
            continue
        if data.shape[0]/2 > frame:
            s, e = int(int(int(data.shape[0])/2) - frame/2), int(int(int(data.shape[0]/2)) + frame/2)
        else:
            s, e = 0, frame
        cliped = data[s:e,:,:].reshape(1,20,256,256)
        if tr_x is None:
            tr_x = cliped
            tr_y = []
            tr_label = np.zeros(2)
            tr_label[label] = 1
            tr_y.append(tr_label)
        else:
            tr_x = np.concatenate([tr_x, cliped])
            tr_label = np.zeros(2)
            tr_label[label] = 1
            tr_y.append(tr_label)
    tr_y = np.array(tr_y)

    for test_label in test_labels:
        id, label = test_label[0], test_label[1]
        vid = f.get(id, None)
        if vid is None:
            nodata.append(id)
            continue
        data = vid['data'][:]
        if data.shape[0] < frame:
            continue
        if data.shape[0] / 2 > frame:
            s, e = int(int(int(data.shape[0]) / 2) - frame / 2), int(int(int(data.shape[0] / 2)) + frame / 2)
        else:
            s, e = 0, frame
        cliped = data[s:e, :, :].reshape(1, 20, 256, 256)
        if te_x is None:
            te_x = cliped
            te_y = []
            te_label = np.zeros(2)
            te_label[label] = 1
            te_y.append(te_label)

        else:
            te_x = np.concatenate([te_x, cliped])

            te_label = np.zeros(2)
            te_label[label] = 1
            te_y.append(te_label)

    te_y = np.array(te_y)

    dataset_tr = MyDataset(tr_x, tr_y, transform=trans)
    train_loader = DataLoader(dataset_tr, batch_size=8, shuffle=True)
    dataset_te = MyDataset(te_x, te_y, transform=trans)
    test_loader = DataLoader(dataset_te, batch_size=8, shuffle=True)
    f.close()
    return train_loader, test_loader

train_loader, test_loader = dataloader(label_file)