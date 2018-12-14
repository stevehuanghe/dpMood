
import os
import pickle
import numpy as np
from datetime import datetime


def load_merged_pickle_data(file_name, mode='short'):
    if not os.path.isfile(file_name):
        print('file not found:', file_name)
    data = pickle.load(open(file_name, 'rb'))
    timestamp = data['timestamp']
    merged = data['merged']
    label = data['label']
    alphanum = data['alphanum']
    accel = data['accel']
    special = data['special']
    merged_s = data['merged_s']
    if mode != 'short':
        merged_s = merged
    merged_100 = []
    alphanum_100 = []
    accel_100 = []
    special_100 = []
    thredhold = 200
    for i in range(len(label)):
        merged_100.append(merged_s[i][:thredhold])
        alphanum_100.append(alphanum[i][:thredhold])
        accel_100.append(accel[i][:thredhold])
        special_100.append(special[i][:thredhold])
    return merged_100, label, timestamp, alphanum_100, accel_100, special_100


def load_pickle_data(file_name):
    if not os.path.isfile(file_name):
        print('file not found:', file_name)
    data, label, timestamp = pickle.load(open(file_name, 'rb'))
    return data, label, timestamp


def split_train_test_data(data_list, train_percent):
    """
    :param data_list: a list of data
    :param train_percent: percentage for training [0,1]
    :return: two lists
    """
    data_length = len(data_list)
    num_train = int(np.ceil(data_length*train_percent))
    perm = np.random.permutation(data_length)
    train_idx = perm[:num_train]
    train_pair = []
    test_pair = []
    for index in range(data_length):
        if index in train_idx:
            train_pair.append(data_list[index])
        else:
            test_pair.append(data_list[index])
    return train_pair, test_pair


def parse_labels(labels, multi_class=False):
    res = []
    for i in range(len(labels)):
        ymrs = (labels[i][1] + 0.0)/30.0
        hdrs = (labels[i][1] + 0.0)/30.0
        res.append((hdrs, ymrs))
    
    return res


def load_data_dir(data_dir, ttype='hour_sum', shuffle=False, multi_class=False, parse_label=True, with_ctrl=True, train_p=0.8, no_split=False):
    """
    :param data_dir: directory that contains the pickle data
    :return: two lists
    """
    files = os.listdir(data_dir)
    file_list = []
    subj_ctrl = [13, 16, 17, 31, 32, 33, 37, 38]
    for filename in files:
        if os.path.isfile(os.path.join(data_dir, filename)):
            name, ext = os.path.splitext(filename)
            if name[0] != '.' and ext == '.pickle':
                if not with_ctrl:
                    sid = int(name.split('_')[1])
                    if sid in subj_ctrl:
                        continue
                file_list.append(filename)
    train_data = []
    test_data = []
    uid = 0
    data_dict = {}
    for filename in file_list:
        data, label, timestamp = load_pickle_data(os.path.join(data_dir, filename))
        accel = data['accel']
        alphanum = data['alphanum']
        special = data['special']
        if parse_label:
            labels = parse_labels(label, multi_class)  # label[0] indicates hdrs, label[1] indicates ymrs
        else:
            labels = label
        timestamp = parse_timestamp(timestamp, ttype=ttype)
        uid_list = [uid for _ in range(len(labels))]
        uid += 1
        data = list(zip(accel, alphanum, special, labels, timestamp, uid_list))
        if no_split:
            data_dict[filename] = data
        if shuffle:
            np.random.shuffle(data)
        index = int(train_p * len(data))
        train_data += data[:index]
        test_data += data[index:]

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    if no_split:
        return data_dict
    return train_data, test_data, uid


def load_subject_data(data_dir):
    files = os.listdir(data_dir)
    file_list = []
    for filename in files:
        if os.path.isfile(os.path.join(data_dir, filename)):
            name, ext = os.path.splitext(filename)
            if name[0] != '.' and ext == '.pickle':
                file_list.append(filename)
    all_data = {}
    uid = 0
    for filename in file_list:
        data, label, timestamp = load_pickle_data(os.path.join(data_dir, filename))
        accel = data['accel']
        alphanum = data['alphanum']
        special = data['special']
        labels = label
        timestamp = parse_timestamp(timestamp, ttype='hour_sum')
        uid_list = [uid for _ in range(len(labels))]
        data = zip(accel, alphanum, special, labels, timestamp, uid_list)
        name, ext = os.path.splitext(filename)
        sid = name.split('_')[1]
        all_data[sid] = data
        uid += 1
    return all_data


def parse_timestamp(timestamp, ttype='hour'):
    res = []
    flag = 'pyd'
    try:
        start_date = timestamp[0].to_pydatetime()
    except:
        flag = 'str'
        start_date = datetime.strptime(timestamp[0], '%Y-%m-%d %H:%M:%S.%f')

    for t in timestamp:
        if flag == 'pyd':
            dt = t.to_pydatetime()
        else:
            dt = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
        if ttype == 'hour':
            td = dt.hour
        elif ttype == 'hour_sum':
            td = int((dt-start_date).total_seconds()/3600)
        elif ttype == 'day_sum':
            td = int((dt - start_date).total_seconds() / (3600*24))
        else:
            td = dt.day
        res.append(td)
    return res


def load_merged_data_dir(data_dir, mode='short', shuffle=False, parse_label=True, with_ctrl=True, train_p=0.8, no_split=False):
    """
    :param data_dir: directory that contains the pickle data
    :return: two lists
    """
    subject_list = [
        'subject_7',
        'subject_13',
        'subject_15',
        'subject_16',
        'subject_17',
        'subject_19',
        'subject_20',
        'subject_21',
        'subject_24',
        'subject_25',
        'subject_27',
        'subject_30',
        'subject_31',
        'subject_32',
        'subject_33',
        'subject_36',
        'subject_37',
        'subject_38',
        'subject_39',
        'subject_40',
    ]
    subj_ctrl = [13, 16, 17, 31, 32, 33, 37, 38]
    files = os.listdir(data_dir)
    file_list = []
    for filename in files:
        if os.path.isfile(os.path.join(data_dir, filename)):
            name, ext = os.path.splitext(filename)
            if name[0] != '.' and ext == '.pickle' and name in subject_list:
                if not with_ctrl:
                    sid = int(name.split('_')[1])
                    if sid in subj_ctrl:
                        continue
                file_list.append(filename)
    train_data = []
    test_data = []
    file_list = sorted(file_list)
    uid = 0
    u_data = {}
    for filename in file_list:
        merged, label, timestamp, alphanum, accel, special = load_merged_pickle_data(os.path.join(data_dir, filename), mode)
        if parse_label:
            labels = parse_labels(label)  # label[0] is hdrs, label[1] is ymrs
        else:
            labels = label
        uid_list = [uid for _ in range(len(labels))]
        uid += 1
        timestamp = parse_timestamp(timestamp, ttype='hour_sum')
        data = zip(merged, labels, timestamp, alphanum, accel, uid_list)
        if no_split:
            u_data[filename] = data
        if shuffle:
            np.random.shuffle(data)
        index = int(train_p * len(data))
        train_data += data[:index]
        test_data += data[index:]
    if no_split:
        return u_data
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    return train_data, test_data


def split_train_test_dict(data_dict, train_p=0.8, shuffle=False, val_p=0):
    train_data = []
    val_data = []
    test_data = []
    for user in data_dict:
        data = data_dict[user]
        if shuffle:
            np.random.shuffle(data)
        index = int(train_p * len(data))
        index2 = int(val_p * len(data))
        train_data += data[:index]
        if val_p > 0:
            val_data += data[index:index2]
            test_data += data[index2:]
        else:
            test_data += data[index:]
    if val_p > 0:
        return train_data, test_data, val_data
    else:
        return train_data, test_data


def save_log_data(data, filename='logs.txt'):
    with open(filename, 'w') as fout:
        for dat in data:
            line = [str(i) for i in dat]
            line = ' '.join(line)
            fout.write(line + '\n')
    return


def load_data_dir_hold2Out(test_ids, data_dir, ttype='hour_sum', shuffle=False, multi_class=False, parse_label=True):
    """
    :param data_dir: directory that contains the pickle data
    :return: two lists
    """
    files = os.listdir(data_dir)
    file_list = []
    for filename in files:
        if os.path.isfile(os.path.join(data_dir, filename)):
            name, ext = os.path.splitext(filename)
            if name[0] != '.' and ext == '.pickle':
                file_list.append(filename)
    train_data = []
    test_data = []
    uid = 0
    for filename in file_list:
        data, label, timestamp = load_pickle_data(os.path.join(data_dir, filename))
        accel = data['accel']
        alphanum = data['alphanum']
        special = data['special']
        if parse_label:
            labels = parse_labels(label, multi_class)  # label[0] indicates hdrs, label[1] indicates ymrs
        else:
            labels = label
        timestamp = parse_timestamp(timestamp, ttype=ttype)
        uid_list = [uid for _ in range(len(labels))]
        uid += 1
        data = zip(accel, alphanum, special, labels, timestamp, uid_list)
        if shuffle:
            np.random.shuffle(data)

        name, ext = os.path.splitext(filename)
        sid = name.split('_')[1]
        if int(sid) in test_ids:
            test_data += data
        else:
            train_data += data

    return train_data, test_data, uid


if __name__ == '__main__':
    #load_merged_data_dir('../data')
    data_dir = '/home/hehuang/Datasets/keyboard'
    files = os.listdir(data_dir)
    print(files)
