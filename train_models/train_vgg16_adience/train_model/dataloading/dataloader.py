import numpy as np
import random


class DataLoader:
    """ manages and provides batches of datasets (train/val)"""

    def __init__(self, dataset, batch_size=32, shuffle=False, startidx=0, endidx=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        if endidx > 0:
            self.idx = list(range(startidx, endidx))
        else:
            self.idx = list(range(len(self.dataset)))

    def __iter__(self):

        if self.shuffle:
            random.shuffle(self.idx)

        self.batches = [self.idx[i:i+self.batch_size] for i in range(0, len(self.idx), self.batch_size)]

        self.current_batch = 0

        return self

    def __next__(self):
        """ Provide next batch. """

        if self.current_batch >= len(self.batches):
            self.epoch_started = False
            raise StopIteration
        else:
            samples = []

            for i in self.batches[self.current_batch]:
                sample = self.dataset[i]
                samples.append(sample)

            self.current_batch += 1

            if self.dataset.mode == "raw":
                return np.array(samples)  # np.array([sample.filename for sample in samples])

            elif self.dataset.mode == "preprocessed":
                return np.array([sample.image for sample in samples]), np.array([sample.one_hot_label for sample in samples])

            elif self.dataset.mode == "binarymap":
                return np.array(samples)

            else:
                return ValueError("dataset mode not valid")
