import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class TaxibjDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset"""

    def __init__(self, X, Y, use_augment=False, data_name='taxibj'):
        super(TaxibjDataset, self).__init__()
        self.X = (X+1) / 2  # channel is 2
        self.Y = (Y+1) / 2
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def _augment_seq(self, seqs):
        """Augmentations as a video sequence"""
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        if self.use_augment:
            len_data = data.shape[0]  # 4
            seqs = self._augment_seq(torch.cat([data, labels], dim=0))
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=None, aft_seq_length=None, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    dataset = np.load(os.path.join(data_root, 'taxibj/dataset.npz'))
    X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset[
        'Y_train'], dataset['X_test'], dataset['Y_test']
    assert X_train.shape[1] == pre_seq_length and Y_train.shape[1] == aft_seq_length
    protocol = os.environ.get(
        'OPENSTL_TAXIBJ_PROTOCOL', 'clean'
    ).strip().lower()

    if protocol == 'official':
        # OpenSTL-compatible protocol:
        # train on all 20,461 training sequences and reuse the
        # official 500-sequence test set for validation and testing.
        train_set = TaxibjDataset(
            X=X_train,
            Y=Y_train,
            use_augment=use_augment,
        )
        test_set = TaxibjDataset(
            X=X_test,
            Y=Y_test,
            use_augment=False,
        )
        vali_set = test_set

    elif protocol == 'clean':
        # Independent chronological validation split.
        total_train = int(X_train.shape[0])
        val_count = max(1, int(round(total_train * 0.10)))
        purge_count = 7
        val_start = total_train - val_count
        train_stop = val_start - purge_count

        assert train_stop > 0

        train_set = TaxibjDataset(
            X=X_train[:train_stop],
            Y=Y_train[:train_stop],
            use_augment=use_augment,
        )
        vali_set = TaxibjDataset(
            X=X_train[val_start:],
            Y=Y_train[val_start:],
            use_augment=False,
        )
        test_set = TaxibjDataset(
            X=X_test,
            Y=Y_test,
            use_augment=False,
        )

    else:
        raise ValueError(
            'OPENSTL_TAXIBJ_PROTOCOL must be clean or official, '
            f'got {protocol!r}'
        )

    print(
        f'TaxiBJ protocol={protocol} '
        f'train={len(train_set)} '
        f'validation={len(vali_set)} '
        f'test={len(test_set)}'
    )

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(vali_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=4)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
