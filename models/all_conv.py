import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class AllConv(nn.Module):
    def __init__(self, use_special=True):
        super(AllConv, self).__init__()
        self.use_special = use_special
        self.conv_accel = nn.Sequential(
            nn.Conv1d(3, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=3, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=3, stride=2),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(11),
        )

        self.conv_alphanum = nn.Sequential(
            nn.Conv1d(4, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=3, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=3, stride=2),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(11),
        )

        self.conv_special = nn.Sequential(
            nn.Conv1d(6, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=3, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=3, stride=2),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(11),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(90, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

        self.fully_connected2 = nn.Sequential(
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

    def forward(self, accel, alphanum, special, timestamp, users):
        accel_feats = self.conv_accel(accel)
        accel_feats = torch.squeeze(accel_feats, dim=2)
        alphanum_feats = self.conv_alphanum(alphanum)
        alphanum_feats = torch.squeeze(alphanum_feats, dim=2)
        special_feats = self.conv_special(special)
        special_feats = torch.squeeze(special_feats, dim=2)

        if self.use_special:
            combined = torch.cat((accel_feats, alphanum_feats, special_feats), 1)
            x = self.fully_connected(combined).view([-1])
        else:
            combined = torch.cat((accel_feats, alphanum_feats), 1)
            x = self.fully_connected2(combined).view([-1])
        return x


if __name__ == '__main__':
    net = AllConv()
    dat = Variable(torch.randn(1, 3, 100))
    res = net.forward(dat)
    print(res.size())


