import contextlib
import os

import torch

class Setting:
    print_shape = False
    save_variable = False
    save_dir = 'dataset/hiddenvec'
    count = 0

@contextlib.contextmanager
def using_config(name, value):
    
    try:
        old_value = getattr(Setting, name)
        setattr(Setting, name, value)
        yield
    finally:
        setattr(Setting, name, old_value)

def saving():
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

        print(f'iteration : {Setting.count} saveing...', end=' ')
        with printing(): vprint(variable, name)
    
        torch.save(variable, f'{os.path.join(temp_dirs, str(Setting.count))}.pt')

@contextlib.contextmanager
def save_counting():
    try:
        yield
    finally:
        Setting.count += 1
        pass
