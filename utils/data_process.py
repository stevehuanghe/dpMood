import csv
import os
import numpy as np
import copy
import pickle
import crash_on_ipy
from datetime import datetime

def load_subject(filename):
    fin = open(filename)
    data = csv.reader(fin)
    cnt = 0
    header = []
    sessions = {}
    for row in data:
        if cnt == 0:
            header = row
        else:
            sess_num = int(float(row[0]))
            # print row[2]
            # raw_input('pause')
            if sess_num not in sessions.keys():
                sessions[sess_num] = {}
                sessions[sess_num]['alphanum'] = []
                sessions[sess_num]['accel'] = []
                sessions[sess_num]['special'] = []
            if row[2] == 'CHAR':
                sessions[sess_num]['alphanum'].append(row[1:])
            elif row[2] == 'ACCELEROMETER_VALUES':
                sessions[sess_num]['accel'].append(row[1:])
            else:
                sessions[sess_num]['special'].append(row[1:])
        cnt += 1
    fin.close()
    return sessions


def load_all_subjects(data_dir):
    files = os.listdir(data_dir)
    file_names = []
    all_subjects_data = {}
    for f in files:
        if f[0] == '.':
            continue
        name, ext = os.path.splitext(f)
        if ext == '.csv' and 'subject' in name:
            print 'loading', name
            all_subjects_data[name] = load_subject(os.path.join(data_dir, f))
    return all_subjects_data


def process_subject(subject, ratings, filename='./data/file.pkl'):
    sessions = subject
    result = {}
    result['label'] = []
    result['merged'] = []
    result['alphanum'] = []
    result['accel'] = []
    result['special'] = []
    result['timestamp'] = []
    result['merged_s'] = []
    max_time = 200
    for key in sessions.keys():
        sess = sessions[key]
        if len(sess['alphanum']) < 10:
            continue

        alphanum = sess['alphanum']
        dat = sess['alphanum'][0][0]
        start_time = datetime.strptime(dat, '%Y-%m-%d %H:%M:%S.%f')
        sess['time'] = dat
        result['timestamp'].append(dat)
        days = start_time.date().toordinal()
        sess['label'] = ratings[days]
        result['label'].append(ratings[days])
        char_normalized = {}

        char_list = []
        accel_list = []
        special_list = []

        idx = 0
        for char in alphanum:
            if idx == 0:
                idx += 1
                char_normalized[0] = np.array([float(x) for x in char[2:5]])
                continue
            tt = datetime.strptime(char[0], '%Y-%m-%d %H:%M:%S.%f')
            dur = (tt - start_time).total_seconds()
            if dur > max_time:
                break
            t_c = int(np.ceil(dur*5))
            if t_c in char_normalized:
                t_c += 1
            #alphanum[idx] = [t_c] + char[1:]
            char_normalized[t_c] = np.array([float(x) for x in char[2:5]])
            #print t_c
            char_list.append([float(x) for x in char[2:5]])
            idx += 1
        sess['alphanum'] = char_normalized
        while len(char_list) <= max_time:
            char_list.append([0, 0, 0])
        result['alphanum'].append(char_list)
        #print alphanum

        accel_normalized = {}
        accel_cnt = {}
        accel = sess['accel']
        for idx in range(len(accel)):
            ac = accel[idx]
            if idx == 0:
                accel_normalized[0] = np.array([float(x) for x in ac[3:6]])
                continue
            tt = datetime.strptime(ac[0], '%Y-%m-%d %H:%M:%S.%f')
            dur = (tt - start_time).total_seconds()
            if dur > max_time:
                break
            t_c = int(np.ceil(dur*5))
            if t_c not in accel_cnt:
                accel_cnt[t_c] = 1.0
                accel_normalized[t_c] = np.array([float(x) for x in ac[3:6]])
                accel_list.append([float(x) for x in ac[3:6]])
            else:
                num = accel_cnt[t_c]
                avg = (accel_normalized[t_c] * num + np.array([float(x) for x in ac[3:6]]))/(num+1)
                accel_cnt[t_c] += 1
                accel_normalized[t_c] = avg
                accel_list.append(list(avg))

        sess['accel'] = accel_normalized
        while len(accel_list) <= max_time:
            accel_list.append([0, 0, 0])
        result['accel'].append(accel_list)

        special_normalized = {}

        special = sess['special']
        for idx in range(len(special)):
            sp = special[idx]
            tt = datetime.strptime(sp[0], '%Y-%m-%d %H:%M:%S.%f')
            dur = (tt - start_time).total_seconds()
            if dur > max_time:
                break
            t_c = int(np.ceil(dur*5))
            if t_c in special_normalized:
                t_c += 1
            special_normalized[t_c] = sp[1]
            special_list.append(sp[1])

        sess['special'] = special_normalized
        while len(accel_list) <= max_time:
            accel_list.append('None')
        result['special'].append(special_list)

        merged = []
        merged_short = []
        for i in range(max_time):
            data_point = []
            if i in sess['accel']:
                data_point += list(sess['accel'][i])
            else:
                data_point += [0, 0, 0]
            if i in sess['alphanum']:
                data_point += list(sess['alphanum'][i])
            else:
                data_point += [0, 0, 0]

            merged.append(data_point)

            if i in sess['alphanum'] and len(merged_short) < 100:
                if i in sess['accel']:
                    dp = list(sess['alphanum'][i]) + list(sess['accel'][i])
                    merged_short.append(dp)
                else:
                    pos = i-1
                    for j in range(i):
                        if pos - j in sess['accel']:
                            dp = list(sess['alphanum'][i]) + list(sess['accel'][pos-j])
                            merged_short.append(dp)
                            break

        sess['merged'] = merged
        result['merged'].append(merged)
        while len(merged_short) <= 100:
            merged_short.append([0, 0, 0, 0, 0, 0])

        result['merged_s'].append(merged_short)
        #import ipdb; ipdb.set_trace()

    with open(filename + '.pickle', 'wb') as fout:
        pickle.dump(result, fout, protocol=pickle.HIGHEST_PROTOCOL)


def parse_ratings(filename=None):
    import csv
    from datetime import datetime
    filename = '/home/hehuang/Datasets/keyboard/raw_data/weekly_ratings.csv'
    fin = open(filename)
    data = csv.reader(fin)

    cnt = 0
    col_hdrs = 0
    col_ymrs = 0
    ratings = {}
    for dat in data:
        if cnt == 0:
            print dat
            col_hdrs = dat.index('sighd_17item')
            col_ymrs = dat.index('ymrs_total')
            print col_hdrs, col_ymrs
            cnt += 1
            continue
        subj_id = int(dat[0])
        if subj_id not in ratings:
            ratings[subj_id] = {}
            ratings[subj_id]['seq'] = []
        try:
            rate_date = datetime.strptime(dat[2], '%Y-%m-%d').date()
        except:
            continue

        rate_day = rate_date.toordinal()

        hdrs_rating = float(dat[col_hdrs])
        ymrs_rating = float(dat[col_ymrs])
        ratings[subj_id][rate_day] = (hdrs_rating, ymrs_rating)

        if len(ratings[subj_id]['seq']) == 0:
            ratings[subj_id]['seq'].append(rate_day)
        elif ratings[subj_id]['seq'][-1] == rate_day:
            continue
        else:
            pre_day = ratings[subj_id]['seq'][-1]
            ratings[subj_id]['seq'].append(rate_day)
            hdrs_frac = float(hdrs_rating - ratings[subj_id][pre_day][0]) / float(rate_day - pre_day)
            ymrs_frac = float(ymrs_rating - ratings[subj_id][pre_day][1]) / float(rate_day - pre_day)
            for i in range(rate_day - pre_day):
                if i == 0:
                    continue
                int_day = rate_day - i
                hdrs_int = hdrs_rating - i * hdrs_frac
                ymrs_int = ymrs_rating - i * ymrs_frac
                ratings[subj_id][int_day] = (hdrs_int, ymrs_int)
                ratings[subj_id]['seq'].append(int_day)
                ratings[subj_id]['seq'] = sorted(ratings[subj_id]['seq'])

    fin.close()
    return ratings


if __name__ == '__main__':
    ratings = parse_ratings()

    all_data = load_all_subjects('/home/hehuang/Datasets/keyboard/raw_data/')
    for subject in all_data:
        print subject
        subj_id = int(subject.split('_')[1])
        process_subject(all_data[subject], ratings[subj_id], '../data/test/' + subject)
        print subject, 'done'

    print 'done'