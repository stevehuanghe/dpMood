import math
import numpy as np


class DataManager(object):
    def __init__(self, data_length, num_epoch, batch_size, shuffle=True):
        self.data_length = data_length
        self.num_epochs = num_epoch
        self.batch_size = batch_size
        self.cur_epoch = 1
        self.cur_batch = 1
        self.cur_pos = 0
        self.num_batch_per_epoch = int(math.ceil(float(data_length)/batch_size))

        self.data_index = []
        res = batch_size - (data_length % batch_size)
        for i in range(num_epoch):
            if shuffle:
                ids = list(np.random.permutation(data_length))
            else:
                ids = [d for d in range(data_length)]
            if res != 0:
                add_on = ids[:res]
                ids.extend(add_on)
            self.data_index.extend(ids)

    def get_batch(self, data):
        start = self.cur_pos
        end = self.cur_pos + self.batch_size - 1
        batch = []
        while start != end + 1:
            batch.append(data[self.data_index[start]])
            start += 1
        self.cur_pos = end + 1

        self.cur_batch += 1
        if (self.cur_batch-1) % self.num_batch_per_epoch == 0:
            self.cur_epoch += 1
            self.cur_batch = 1
        return batch


if __name__ == '__main__':
    dm = DataManager(12, 2, 5)
    data = [i for i in range(12)]
    for i in range(dm.num_epochs):
        for j in range(dm.num_batch_per_epoch):
            print('epoch: %d, batch: %d'%(dm.cur_epoch, dm.cur_batch))
            batch = dm.get_batch(data)
            print(batch)
