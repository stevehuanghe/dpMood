import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models.cnn_rnn import CNN_RNN, RNN, CNN_RNN_1, CNN_RNN_EF2_PS
from models.all_conv import AllConv

from utils.utils import *
from utils.data_manager import DataManager

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

result_dir = './results_all/'
data_dir = '/home/hehuang/Datasets/keyboard'
merged_data_dir = '../keyboard_cnn/data'
checkpoint_dir = './checkpoints/'

num_runs = 20
batch_size = 256
num_epochs = 200
learning_rate = 1e-3

mode = 'rgs_hdrs'
use_ctrl = False

parse_label = False

model_list = ['2cnn_rnn_sin_id', '2cnn_rnn_sin', '2cnn_rnn', 'cnn', 'rnn']

if use_ctrl:
    ctrl = ''
else:
    ctrl = '_NoCtrl'


if not os.path.isdir(result_dir):
    os.system('mkdir ' + result_dir)

print 'loading data...'
train_data_d, test_data_d, num_users = load_data_dir(data_dir=data_dir, ttype='hour_sum', shuffle=False,
                                                     multi_class=False, parse_label=parse_label, with_ctrl=use_ctrl)

log_file = './logs/log_lf_stats_' + mode + ctrl + '.txt'


def main(model='2cnn_rnn_sin_id', pid=0):
    if parse_label:
        pl = '_norm'
    else:
        pl = ''
    pl += ('_' + str(pid))
    checkpoint_path = checkpoint_dir + model + '_' + mode + ctrl + pl + '.ckpt'
    if model == 'DeeperMood_24h':
        log_file = './results/log_DM_24h2_' + mode + ctrl + pl + '.txt'
        ab_log = './results/hours_emb_2_' + mode + ctrl + pl + '.txt'
        net = CNN_RNN(use_time='24h2')
    elif model == '2cnn_rnn_sin_id':
        log_file = result_dir + 'log_2cnn_rnn_sin_id_' + mode + ctrl + pl + '.txt'
        ab_log = result_dir + 'alpha_beta_id_' + mode + ctrl + pl + '.txt'
        net = CNN_RNN(use_time='sin_id', num_user=num_users)
    elif model == '2cnn_rnn_sin':
        log_file = result_dir + 'log_2cnn_rnn_sin_' + mode + ctrl + pl + '.txt'
        net = CNN_RNN(use_time='sin', num_user=num_users)
    elif model == '2cnn_rnn':
        log_file = result_dir + 'log_2cnn_rnn_' + mode + ctrl + pl + '.txt'
        net = CNN_RNN(use_time=None)
    elif model == 'cnn_rnn':
        log_file = result_dir + 'log_cnn_rnn_' + mode + ctrl + pl + '.txt'
        net = CNN_RNN_1()
    elif model == 'cnn':
        log_file = result_dir + 'log_cnn_' + mode + ctrl + pl + '.txt'
        net = AllConv()
    elif model == 'cnn2':
        log_file = result_dir + 'log_cnn2_' + mode + ctrl + pl + '.txt'
        net = AllConv(use_special=False)
    else: # model == 'rnn':
        log_file = result_dir + 'log_rnn_' + mode + ctrl + pl + '.txt'
        net = RNN()

    train_data = train_data_d
    test_data = test_data_d

    net.cuda()
    if mode == 'clf':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=0.001)
    print 'training...'

    best_epoch = 0
    best_acc = 0
    best_f1 = 0
    best_rmse = 1e10
    results = []
    train_data_manager = DataManager(len(train_data), num_epoch=num_epochs, batch_size=batch_size)

    for epoch in range(num_epochs):
        num_batch = train_data_manager.num_batch_per_epoch
        net.train()
        for batch in range(num_batch):
            optimizer.zero_grad()

            t0 = time.time()
            batch_data = train_data_manager.get_batch(train_data)
            accel = Variable(torch.from_numpy(np.transpose(np.asarray([item[0] for item in batch_data]), [0, 2, 1])).cuda())
            alphanum = Variable(torch.from_numpy(np.transpose(np.asarray([item[1] for item in batch_data]), [0, 2, 1])).cuda())
            special = Variable(torch.from_numpy(np.transpose(np.asarray([item[2] for item in batch_data]), [0, 2, 1])).cuda())
            timestamp = Variable(torch.from_numpy(np.asarray([item[4] for item in batch_data])).cuda().float())
            users = Variable(torch.from_numpy(np.asarray([item[5] for item in batch_data])).cuda())
            labels = [item[3] for item in batch_data]
            label_HDRS = Variable(torch.from_numpy(np.asarray([item[0] for item in labels])).cuda()).float()
            label_YMRS = Variable(torch.from_numpy(np.asarray([item[1] for item in labels])).cuda()).float()
            t1 = time.time()

            outputs = net.forward(accel, alphanum, special, timestamp, users)

            if mode == 'clf' or mode == 'rgs_hdrs':
                loss = criterion(outputs, label_HDRS)
            else:
                loss = criterion(outputs, label_YMRS)

            loss.backward()
            optimizer.step()
            t2 = time.time()

        res = evaluate(test_data, net)
        results.append(res)

        if res[0] < best_rmse:
            best_rmse = res[0]
            best_epoch = epoch + 1
            #torch.save(net.state_dict(), checkpoint_path)
        #print 'best: epoch %d, rmse: %f' % (best_epoch, best_rmse)

    if mode == 'clf':
        print 'best: epoch %d, acc: %f, f1: %f'%(best_epoch, best_acc, best_f1)
    else:
        print 'best: epoch %d, rmse: %f'%(best_epoch, best_rmse)

    save_log_data(results, log_file)

    if model == '2cnn_rnn_sin_id':
        idxs = Variable(torch.from_numpy(np.asarray([i for i in range(num_users)])).long().cuda())
        alpha = net.alpha(idxs).data
        beta = net.beta(idxs).data
        gamma = net.gamma(idxs).data
        delta = net.delta(idxs).data
        with open(ab_log, 'w') as fout:
            for i in range(num_users):
                fout.write(str(alpha[i][0]) + '\t' + str(beta[i][0]) + '\t' + str(gamma[i][0]) + '\t' + str(delta[i][0]) + '\n')

    return best_rmse


def evaluate(test_data, net, filepath=None):
    net.eval()
    test_data_manager = DataManager(len(test_data), num_epoch=1, batch_size=batch_size)
    num_batch = test_data_manager.num_batch_per_epoch
    pred = []
    truth = []
    logits = []
    for batch in range(num_batch):
        t0 = time.time()
        batch_data = test_data_manager.get_batch(test_data)
        accel = Variable(torch.from_numpy(np.transpose(np.asarray([item[0] for item in batch_data]), [0, 2, 1])).cuda())
        alphanum = Variable(
            torch.from_numpy(np.transpose(np.asarray([item[1] for item in batch_data]), [0, 2, 1])).cuda())
        special = Variable(
            torch.from_numpy(np.transpose(np.asarray([item[2] for item in batch_data]), [0, 2, 1])).cuda())
        labels = [item[3] for item in batch_data]
        label_HDRS = [item[0] for item in labels]
        label_YMRS = [item[1] for item in labels]

        timestamp = Variable(torch.from_numpy(np.asarray([item[4] for item in batch_data])).cuda().float())
        users = Variable(torch.from_numpy(np.asarray([item[5] for item in batch_data])).cuda())

        t1 = time.time()
        outputs = net.forward(accel, alphanum, special, timestamp, users)
        if mode == 'clf':
            probs = torch.sigmoid(outputs)
            logits += list(probs.data.cpu().numpy())
            pred += list(probs.data.cpu().numpy() > 0.5)
            truth += label_HDRS
        else:
            pred += list(outputs.data.cpu().numpy())
            if mode == 'rgs_hdrs':
                truth += label_HDRS
            else:
                truth += label_YMRS

    if mode == 'clf':
        acc = accuracy_score(y_true=truth, y_pred=pred)
        f1 = f1_score(y_true=truth, y_pred=pred)
        if filepath is not None:
            save_log_data([logits, truth], filepath)
        return [acc, f1]
    else:
        rmse = np.sqrt(mean_squared_error(y_true=truth, y_pred=pred))
        return [rmse]


if __name__ == '__main__':
    import gc
    from utils.logger import *
    logger = Logger(filename=log_file).get_logger()
    #torch.manual_seed(1234)
    #torch.cuda.manual_seed_all(1234)
    for m in model_list:
        rmse_list = []
        logger.info('running model: ' + m)
        for n in range(num_runs):
            rmse = main(model=m, pid=n)
            rmse_list.append(rmse)
            logger.info('run %d, rmse %f'%(n, rmse))
        rmse_mean = float(np.mean(rmse_list))
        rmse_max = max(rmse_list)
        rmse_min = min(rmse_list)
        rmse_std = float(np.std(rmse_list))
        logger.info('mean: %f, std: %f, max: %f, min: %f'%(rmse_mean, rmse_std, rmse_max, rmse_min))
        gc.collect()



