import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class CNN_RNN(nn.Module):
    def __init__(self, use_time=None, num_user=20, num_sin=1):
        super(CNN_RNN, self).__init__()
        self.use_time = use_time
        self.num_user = num_user
        self.num_sin = num_sin
        self.z = 3.1415926/12
        if use_time in ['sin_id', 'sin_id_0', 'sin_id_fix', 'sin_id_bd', 'only_c']:
            self.alpha = nn.Embedding(num_user, num_sin)
            self.beta = nn.Embedding(num_user, num_sin)
            self.gamma = nn.Embedding(num_user, num_sin)
            self.delta = nn.Embedding(num_user, num_sin)
        elif use_time == '24h':
            self.hour_embeddings = nn.Embedding(24, 10)
        elif use_time == '24h2':
            self.hour_embeddings = nn.Embedding(24, 1)
        else:
            self.alpha = Variable(torch.FloatTensor(1)).cuda()
            torch.nn.init.constant(self.alpha, 1)
            self.beta = Variable(torch.FloatTensor(1)).cuda()
            torch.nn.init.constant(self.beta, 1)
            self.gamma = Variable(torch.FloatTensor(1)).cuda()
            torch.nn.init.constant(self.gamma, 1)
            self.delta = Variable(torch.FloatTensor(1)).cuda()
            torch.nn.init.constant(self.delta, 1)

        self.conv_accel = nn.Sequential(
            nn.Conv1d(3, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )

        self.conv_alphanum = nn.Sequential(
            nn.Conv1d(4, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )
        self.conv_special = nn.Sequential(
            nn.Conv1d(6, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )
        feat_dim = 20
        self.rnn_accel = nn.GRU(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)
        self.rnn_alphanum = nn.GRU(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)
        self.rnn_special = nn.GRU(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)

        self.linear = nn.Sequential(
            nn.Linear(120, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        self.linearHr24 = nn.Sequential(
            nn.Linear(90, 45),
            nn.ReLU(),
            nn.Linear(45, 1)
        )

        self.linearMC = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 5)
        )

    def forward(self, accel, alphanum, special, timestamp=None, users=None):

        accel_feats = self.conv_accel(accel)
        accel_feats = torch.transpose(accel_feats, 1, 2)
        _, accel_feats = self.rnn_accel(accel_feats)
        accel_feats = torch.cat((accel_feats[0], accel_feats[1]), 1)

        alphanum_feats = self.conv_alphanum(alphanum)
        alphanum_feats = torch.transpose(alphanum_feats, 1, 2)
        _, alphanum_feats = self.rnn_alphanum(alphanum_feats)
        alphanum_feats = torch.cat((alphanum_feats[0], alphanum_feats[1]), 1)

        # special_feats = self.conv_special(special)
        # special_feats = torch.transpose(special_feats, 1, 2)
        # _, special_feats = self.rnn_special(special_feats)
        # special_feats = torch.cat((special_feats[0], special_feats[1]), 1)

        #x = torch.cat((accel_feats, alphanum_feats, special_feats), 1)
        if self.use_time == '24h':
            time_feats = self.hour_embeddings(timestamp)
            x = torch.cat((accel_feats, alphanum_feats, time_feats), 1)
            x = self.linearHr24(x).view([-1])
        elif self.use_time == '24h2':
            time_feats = self.hour_embeddings(timestamp).view([-1])
            x = torch.cat((accel_feats, alphanum_feats), 1)
            x = self.linear2(x).view([-1])
            x = x * time_feats
        else:
            x = torch.cat((accel_feats, alphanum_feats), 1)
            x = self.linear2(x).view([-1])

        if self.use_time == 'sin_id':
            u_alpha = self.alpha(users).view([-1])
            u_beta = self.beta(users).view([-1])
            u_gamma = self.gamma(users).view([-1])
            u_delta = self.delta(users).view([-1])
            x = x * (u_gamma * torch.sin(u_alpha * timestamp + u_beta) + u_delta)
        elif self.use_time == 'sin':
            x = x * (self.gamma * torch.sin(self.alpha * timestamp + self.beta) + self.delta)
        elif self.use_time == 'sin_id_0':
            u_alpha = self.alpha(users).view([-1])
            u_beta = self.beta(users).view([-1])
            x = torch.sin(u_alpha * x + u_beta)
        elif self.use_time == 'sin_id_fix':
            u_beta = self.beta(users).view([-1])
            u_gamma = self.gamma(users).view([-1])
            u_delta = self.delta(users).view([-1])
            x = x * (u_gamma * torch.sin(self.z * timestamp + u_beta) + u_delta)
        elif self.use_time == 'sin_id_bd':
            u_beta = self.beta(users).view([-1])
            u_delta = self.delta(users).view([-1])
            x = x * (torch.sin(self.z * timestamp + u_beta) + u_delta)
        elif self.use_time == 'only_c':
            u_delta = self.delta(users).view([-1])
            x = x * u_delta

        return x

    def test_infer(self,  accel, alphanum, special, timestamp):
        accel_feats = self.conv_accel(accel)
        accel_feats = torch.transpose(accel_feats, 1, 2)
        _, accel_feats = self.rnn_accel(accel_feats)
        accel_feats = torch.cat((accel_feats[0], accel_feats[1]), 1)

        alphanum_feats = self.conv_alphanum(alphanum)
        alphanum_feats = torch.transpose(alphanum_feats, 1, 2)
        _, alphanum_feats = self.rnn_alphanum(alphanum_feats)
        alphanum_feats = torch.cat((alphanum_feats[0], alphanum_feats[1]), 1)

        special_feats = self.conv_special(special)
        special_feats = torch.transpose(special_feats, 1, 2)
        _, special_feats = self.rnn_special(special_feats)
        special_feats = torch.cat((special_feats[0], special_feats[1]), 1)

        x = torch.cat((accel_feats, alphanum_feats), 1)
        x = self.linear2(x).view([-1])

        u_alpha = torch.mean(self.alpha.weight.view([-1]))
        u_beta = torch.mean(self.beta.weight.view([-1]))
        u_gamma = torch.mean(self.gamma.weight.view([-1]))
        u_delta = torch.mean(self.delta.weight.view([-1]))
        x = x * (u_gamma * torch.sin(u_alpha * timestamp + u_beta) + u_delta)

        return x

    def get_feature(self, accel, alphanum, special, timestamp, users):
        accel_feats = self.conv_accel(accel)
        accel_feats = torch.transpose(accel_feats, 1, 2)
        _, accel_feats = self.rnn_accel(accel_feats)
        accel_feats = torch.cat((accel_feats[0], accel_feats[1]), 1)

        alphanum_feats = self.conv_alphanum(alphanum)
        alphanum_feats = torch.transpose(alphanum_feats, 1, 2)
        _, alphanum_feats = self.rnn_alphanum(alphanum_feats)
        alphanum_feats = torch.cat((alphanum_feats[0], alphanum_feats[1]), 1)

        x = torch.cat((accel_feats, alphanum_feats), 1)
        return x


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        feat_dim = 15
        self.rnn_accel = nn.GRU(3, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)
        self.rnn_alphanum = nn.GRU(4, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)
        self.rnn_special = nn.GRU(6, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)

        self.linear = nn.Sequential(
            nn.Linear(90, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

    def forward(self, accel, alphanum, special, timestamp=None, users=None):
        accel_feats = torch.transpose(accel, 1, 2)
        _, accel_feats = self.rnn_accel(accel_feats)
        accel_feats = torch.cat((accel_feats[0], accel_feats[1]), 1)

        alphanum_feats = torch.transpose(alphanum, 1, 2)
        _, alphanum_feats = self.rnn_alphanum(alphanum_feats)
        alphanum_feats = torch.cat((alphanum_feats[0], alphanum_feats[1]), 1)

        special_feats = torch.transpose(special, 1, 2)
        _, special_feats = self.rnn_special(special_feats)
        special_feats = torch.cat((special_feats[0], special_feats[1]), 1)

        x = torch.cat((accel_feats, alphanum_feats, special_feats), 1)

        x = self.linear(x).view([-1])
        return x


class CNN_RNN_1(nn.Module):
    def __init__(self):
        super(CNN_RNN_1, self).__init__()

        self.conv_accel = nn.Sequential(
            nn.Conv1d(3, 20, kernel_size=3, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )

        self.conv_alphanum = nn.Sequential(
            nn.Conv1d(4, 20, kernel_size=3, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )
        self.conv_special = nn.Sequential(
            nn.Conv1d(6, 20, kernel_size=3, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )
        feat_dim = 20
        self.rnn_accel = nn.GRU(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)
        self.rnn_alphanum = nn.GRU(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)
        self.rnn_special = nn.GRU(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)

        self.linear = nn.Sequential(
            nn.Linear(120, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, accel, alphanum, special, timestamp, users):

        accel_feats = self.conv_accel(accel)
        accel_feats = torch.transpose(accel_feats, 1, 2)
        _, accel_feats = self.rnn_accel(accel_feats)
        accel_feats = torch.cat((accel_feats[0], accel_feats[1]), 1)

        alphanum_feats = self.conv_alphanum(alphanum)
        alphanum_feats = torch.transpose(alphanum_feats, 1, 2)
        _, alphanum_feats = self.rnn_alphanum(alphanum_feats)
        alphanum_feats = torch.cat((alphanum_feats[0], alphanum_feats[1]), 1)

        special_feats = self.conv_special(special)
        special_feats = torch.transpose(special_feats, 1, 2)
        _, special_feats = self.rnn_special(special_feats)
        special_feats = torch.cat((special_feats[0], special_feats[1]), 1)

        #x = torch.cat((accel_feats, alphanum_feats, special_feats), 1)
        x = torch.cat((accel_feats, alphanum_feats), 1)
        #x = accel_feats

        x = self.linear2(x).view([-1])

        return x


class CNN_RNN_EF(nn.Module):
    def __init__(self, use_accel=True):
        super(CNN_RNN_EF, self).__init__()
        self.use_accel = use_accel
        self.conv = nn.Sequential(
            nn.Conv1d(6, 20, kernel_size=3, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Conv1d(20, 40, kernel_size=5, stride=2),
            nn.BatchNorm1d(40),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(40, 40, 1, batch_first=True, bidirectional=True, dropout=0.1)

        self.linear = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, x):
        conv_feats = self.conv(x)
        _, rnn_feats = self.rnn(torch.transpose(conv_feats, 2, 1))
        rnn_feats = torch.cat((rnn_feats[0], rnn_feats[1]), 1)
        res = self.linear(rnn_feats).view([-1])
        return res

# early fusion that ignores missing values, and use accelrometer
class CNN_RNN_EF2(nn.Module):
    def __init__(self, use_accel=True):
        super(CNN_RNN_EF2, self).__init__()
        self.use_accel = use_accel
        self.conv_accel = nn.Sequential(
            nn.Conv1d(3, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )

        self.conv_alphanum = nn.Sequential(
            nn.Conv1d(6, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )

        feat_dim = 20
        self.rnn_accel = nn.GRU(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)
        self.rnn_alphanum = nn.GRU(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)

        self.linear = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, alphanum, accel, timestamp=None):

        accel_feats = self.conv_accel(accel)
        accel_feats = torch.transpose(accel_feats, 1, 2)
        _, accel_feats = self.rnn_accel(accel_feats)
        accel_feats = torch.cat((accel_feats[0], accel_feats[1]), 1)

        alphanum_feats = self.conv_alphanum(alphanum)
        alphanum_feats = torch.transpose(alphanum_feats, 1, 2)
        _, alphanum_feats = self.rnn_alphanum(alphanum_feats)
        alphanum_feats = torch.cat((alphanum_feats[0], alphanum_feats[1]), 1)

        if self.use_accel:
            x = torch.cat((accel_feats, alphanum_feats), 1)
            x = self.linear(x).view([-1])
        else:
            x = self.linear2(alphanum_feats).view([-1])
        return x


# early fusion that ignores missing values, and use accelrometer
class CNN_RNN_EF2_PS(nn.Module):
    def __init__(self, use_accel=True, num_user=20):
        super(CNN_RNN_EF2_PS, self).__init__()
        self.use_accel = use_accel
        self.num_user = num_user
        self.alpha = nn.Embedding(num_user, 1)
        self.beta = nn.Embedding(num_user, 1)
        self.gamma = nn.Embedding(num_user, 1)
        self.delta = nn.Embedding(num_user, 1)
        self.conv_accel = nn.Sequential(
            nn.Conv1d(3, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )

        self.conv_alphanum = nn.Sequential(
            nn.Conv1d(6, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )

        feat_dim = 20
        self.rnn_accel = nn.GRU(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)
        self.rnn_alphanum = nn.GRU(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True, dropout=0.1)

        self.linear = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, alphanum, accel, timestamp=None, users=None):

        accel_feats = self.conv_accel(accel)
        accel_feats = torch.transpose(accel_feats, 1, 2)
        _, accel_feats = self.rnn_accel(accel_feats)
        accel_feats = torch.cat((accel_feats[0], accel_feats[1]), 1)

        alphanum_feats = self.conv_alphanum(alphanum)
        alphanum_feats = torch.transpose(alphanum_feats, 1, 2)
        _, alphanum_feats = self.rnn_alphanum(alphanum_feats)
        alphanum_feats = torch.cat((alphanum_feats[0], alphanum_feats[1]), 1)

        if self.use_accel:
            x = torch.cat((accel_feats, alphanum_feats), 1)
            x = self.linear(x).view([-1])
        else:
            x = self.linear2(alphanum_feats).view([-1])

        u_alpha = self.alpha(users).view([-1])
        u_beta = self.beta(users).view([-1])
        u_gamma = self.gamma(users).view([-1])
        u_delta = self.delta(users).view([-1])
        x = x * (u_gamma * torch.sin(u_alpha * timestamp + u_beta) + u_delta)

        return x



