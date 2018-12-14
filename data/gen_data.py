import numpy as np
import pickle as pk
import datetime

def gen_synthetic_data(num_users, num_sessions):
    for uid in range(num_users):
        accel = np.random.randn(num_sessions, 100, 3)
        alpha = np.random.randn(num_sessions, 100, 4)
        special = np.random.randn(num_sessions, 100, 6)
        timestamp = [str(datetime.datetime.now()) for _ in range(num_sessions)]
        labels = np.random.uniform(0, 30, [num_sessions, 2])
        data = {
            'accel': accel,
            'alphanum': alpha,
            'special': special
        }
        filename = "subject_" + str(uid) + ".pickle"
        with open(filename, 'wb') as fout:
            pk.dump([data, labels, timestamp], fout)

if __name__ == "__main__":            
	gen_synthetic_data(3, 10)
