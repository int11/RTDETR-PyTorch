import contextlib
import os

import torch


class Setting:
    print_shape = False
    save_variable = False
    dirs = 'dataset/hiddenvec'

def vprint(name, variable):
    """
    if print_shape is True, print the shape of the variable
    """
    if Setting.print_shape:
        if isinstance(variable, list):
            print(f"{name} shape: {[i.shape for i in variable]}")
        else:
            print(f"{name} shape: {variable.shape}")


def save(variable):
    if Setting.save_variable:
        if not os.path.exists(Setting.dirs):
            os.makedirs(Setting.dirs)

        print(f'saving {variable.shape}')
        torch.save(variable, 'dataset/hiddenvec/variable.pt')
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
    old_value = Setting.save_variable
    Setting.save_variable = True
    try:
        yield
    finally:
        Setting.save_variable = old_value
