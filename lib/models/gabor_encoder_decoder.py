import torch
import torch.nn as nn
from torch.nn import functional as F
from GaborNet import GaborConv2d
from torch.autograd import Variable
import cv2
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_filename = "./2.ppm"
image = cv2.imread(input_filename)
image = np.moveaxis(image, -1, 0)[None, ...]
image = torch.from_numpy(image).cuda().float()

class GaborNN(nn.Module):
    def __init__(self):
        super(GaborNN, self).__init__()
        self.g0 = GaborConv2d(in_channels=3, out_channels=96, kernel_size=(11, 11))
        self.c1 = nn.Conv2d(96, 384, (3, 3))
        self.fcs = nn.Sequential(
            nn.Linear(384 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 96),
        )

    def forward(self, x):
        x = F.leaky_relu(self.g0(x))
        x = nn.MaxPool2d(kernel_size=3)(x)
        x = F.leaky_relu(self.c1(x))
        x = nn.MaxPool2d(kernel_size=11)(x)
        x = x.view(-1, 384 * 500)
        x = self.fcs(x)
        return x


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=96, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=3):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

