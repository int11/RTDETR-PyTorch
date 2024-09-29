
import time
from contextlib import contextmanager

import torch
from torch import tensor

from function import attentionWeight_twice_matmul_type1, attentionWeight_twice_matmul_type1_c, attentionWeight_twice_matmul_type2, attentionWeight_twice_matmul_type3


@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"[{name}] elapsed time: {end - start:.4f} seconds")

original_shape = (16, 16)

weight_shape = 4, 256, 256
w = torch.arange(1, torch.prod(tensor(weight_shape))+1, dtype=torch.float32).reshape(weight_shape) * 0.1

shape = (4, 255, 64, 64)
a = torch.arange(1, torch.prod(tensor(shape))+1, dtype=torch.float32).reshape(shape) * 0.1


with timer("attentionWeight_twice_matmul_type1"):
    r1 = attentionWeight_twice_matmul_type1(w, a, 4)
    # print(r1)

with timer("attentionWeight_twice_matmul_type1_c"):
    r2 = attentionWeight_twice_matmul_type1_c(w, a, 4)
    # print(r2)

with timer("attentionWeight_twice_matmul_type2"):
    r3 = attentionWeight_twice_matmul_type2(w, a, original_shape)
    # print(r3)

with timer("attentionWeight_twice_matmul_type3"):
    r4 = attentionWeight_twice_matmul_type3(w, a, original_shape)
    # print(r4)