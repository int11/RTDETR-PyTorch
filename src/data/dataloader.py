import torch 
import torch.utils.data as data


__all__ = ['DataLoader']


def default_collate_fn(items):
    '''default collate_fn
    '''    
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class DataLoader(data.DataLoader):
    def __init__(
            self, 
            dataset,
            batch_size,
            shuffle=True, 
            num_workers=0,
            collate_fn=default_collate_fn, 
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