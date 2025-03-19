"""
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.utils.data as data
import torch.nn.functional as F
import random


class DataLoader(data.DataLoader):
    def __init__(
            self, 
            dataset,
            batch_size,
            shuffle=True, 
            num_workers=0,
            collate_fn=None, 
            drop_last=True, 
            **kwargs):
        
        self.shuffle = shuffle
        super(DataLoader, self).__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            collate_fn=collate_fn, 
            drop_last=drop_last, 
            **kwargs)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch 
        self.collate_fn.set_epoch(epoch)
    
    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


class BaseCollateFunction(object):
    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1
    
    def set_epoch(self, epoch):
        self._epoch = epoch 

    def __call__(self, items):
        raise NotImplementedError('')


class BatchImageCollateFuncion(BaseCollateFunction):
    def __init__(
        self, 
        scales=None, 
        stop_epoch=None, 
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        # self.interpolation = interpolation

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        return images, targets

