from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler, Sampler
import logging
logger = logging.getLogger("__main__.data_loader")


def build_sampler(shuffle, dataset):
    if shuffle:
        return RandomSampler(dataset)
    else:
        return SequentialSampler(dataset)


class LoopBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size,drop_last=False, loop=False):
        self.loop = loop

        super(LoopBatchSampler, self).__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        count = 0
        while True:
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    if isinstance(batch[0], list) and self.batch_size == 1:
                        batch = batch[0]
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
            count += 1
            if not self.loop:
                raise StopIteration

    def __len__(self):
        return len(self.sampler)


def build_data_loader(dataset, loop=False, shuffle=False,
                      batch_size=4, drop_last=False, num_workers=0):
    sampler = build_sampler(shuffle, dataset)
    batch_sampler = LoopBatchSampler(sampler,
                                     batch_size=batch_size,
                                     drop_last=drop_last,
                                     loop=loop)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                             num_workers=num_workers, pin_memory=True)
    return data_loader


class TestDataset(Dataset):
    def __init__(self, n=10):
        self.data = list(range(n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx


def print_data(dataloader):
    for i, data in enumerate(dataloader):
        print("batch {} : data {}".format(i, data))


if __name__=='__main__':
    dataset = TestDataset()
    kwargs = dict(
        dataset=dataset,
        loop=True,
        shuffle=False,
        batch_size=4,
        drop_last=False,
    )
    dataloader = build_data_loader(**kwargs)
    data_iterator = iter(dataloader)
    for i in range(10):
        data = next(data_iterator)
        print("Loop batch {} : data {}".format(i, data))

    kwargs = dict(
        dataset=dataset,
        loop=False,
        shuffle=False,
        batch_size=4,
        drop_last=False,
    )
    dataloader = build_data_loader(**kwargs)
    for i, data in enumerate(dataloader):
        print("Not loop batch {} : data {}".format(i, data))






