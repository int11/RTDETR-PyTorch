import contextlib
import os

import torch


class Setting:
    print_shape = False
    save_variable = False
    save_dir = 'dataset/hiddenvec'
    count = 0

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

        vprint(variable, name)
        
        torch.save(variable, f'{os.path.join(temp_dirs, str(Setting.count))}.pt')
        print(f'saved')

@contextlib.contextmanager
def saving():
    """
    this function is used to save the variable with python "with" statement
    example:

    with saving():
        # do something
        # The save function in it save all variables.
    """
    try:
        old_value = Setting.save_variable
        Setting.save_variable = True
        yield
    finally:
        Setting.save_variable = old_value

@contextlib.contextmanager
def save_counting():
    try:
        yield
    finally:
        Setting.count += 1
        pass
