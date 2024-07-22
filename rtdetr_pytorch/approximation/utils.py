import contextlib
import os

import torch

class Setting:
    print_shape = False
    save_variable = False
    save_dir = 'dataset/hiddenvec'
    index = []


class vechook:
    hooking = False
    variable = {}
    
    def __enter__(self):
        vechook.hooking = True
        return self
    
    @classmethod
    def hook(cls, variable, name):
        if cls.hooking == False:
            return
        cls.variable[name] = variable
    
    def __exit__(self, *exc_details):
        vechook.hooking = False
        return self
    
@contextlib.contextmanager
def using_config(name, value):
    try:
        old_value = getattr(Setting, name)
        setattr(Setting, name, value)
        yield
    finally:
        setattr(Setting, name, old_value)

def saving(index):
    Setting.index = index
    return using_config('save_variable', True)

def printing():
    return using_config('print_shape', True)

def vprint(variable, name):
    """
    if print_shape is True, print the shape of the variable
    """
    if Setting.print_shape:
        if isinstance(variable, list):
            print(f"{name} shape: {[i.shape for i in variable]}")
        else:
            print(f"{name} shape: {variable.shape}")

def save(variable, name):
    temp_dirs = os.path.join(Setting.save_dir, name)

    if Setting.save_variable:
        if not os.path.exists(temp_dirs):
            os.makedirs(temp_dirs)

        print('Saving... ', end='')
        with printing(): vprint(variable, name)
        
        for i in range(len(Setting.index)):
            a = [e[i] for e in variable]

            torch.save(a, f'{os.path.join(temp_dirs, str(Setting.index[i]))}.pt')
