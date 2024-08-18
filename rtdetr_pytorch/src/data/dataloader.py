import torch 
import torch.utils.data as data

from src.core import register


__all__ = ['DataLoader']


@register
class DataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.shuffle = kwargs.get('shuffle', False)
        kwargs['shuffle'] = self.shuffle
        super(DataLoader, self).__init__(*args, **kwargs)
        

    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string



@register
def default_collate_fn(items):
    '''default collate_fn
    '''    
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]
