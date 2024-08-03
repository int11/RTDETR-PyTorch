import contextlib
import os

import torch
import time

try:
    if 'root' in os.path.expanduser('~'):
        cache_dir = '/content/drive/MyDrive/.RT-DETR'
    else:
        cache_dir = os.path.join(os.path.expanduser('~'), '.RT-DETR')
except:
    print("No cache dir found to store weights.")

class Setting:
    print_shape = False
    save_variable = False
    save_dir = 'dataset/hiddenvec'
    index = []


class vechook:
    vechook_instances = {}
    hooking = False
    
    def __init__(self, name='defualt') -> None:
        vechook.vechook_instances[name] = self
        self.name = name
        self.variable = {}
        
    def __enter__(self):
        vechook.hooking = True
        return self
    
    @classmethod
    def hook(cls, variables, names, instance_name='defualt'):
        if cls.hooking == False:
            return
        
        if isinstance(variables, tuple):
            variables = (variables)
        if isinstance(names, tuple):
            names = (names)

        instance = cls.vechook_instances[instance_name]

        for variable, name in zip(variables, names):
            instance.variable[name] = variable
    
    @classmethod
    def hookappend(cls, variables, names, instance_name='defualt'):
        if cls.hooking == False:
            return
        
        if isinstance(variables, tuple):
            variables = (variables)
        if isinstance(names, tuple):
            names = (names)

        instance = cls.vechook_instances[instance_name]

        for variable, name in zip(variables, names):
            if name not in instance.variable:
                instance.variable[name] = []
            instance.variable[name].append(variable)

    def __exit__(self, *exc_details):
        vechook.hooking = False
        del vechook.vechook_instances[self.name]
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


class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *exc_details):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.name} elapsed time: {self.elapsed_time:.4f} sec")