
import time
from contextlib import contextmanager

import torch
from torch import tensor

from function import attentionWeight_twice_matmul_type1, attentionWeight_twice_matmul_type1_c, attentionWeight_twice_matmul_type2, attentionWeight_twice_matmul_type3, attentionWeight_twice_matmul_type3_c


@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"[{name}] elapsed time: {end - start:.4f} seconds")

img_H_W = (16, 16)
scale = 4
batch_size = 50
attention_weight_shape = (batch_size, img_H_W[0] ** 2, img_H_W[1] ** 2)
w = torch.arange(1, torch.prod(tensor(attention_weight_shape))+1, dtype=torch.float32).reshape(attention_weight_shape) * 0.1

shape = (batch_size, 255, img_H_W[0] * scale, img_H_W[1] * scale)
a = torch.arange(1, torch.prod(tensor(shape))+1, dtype=torch.float32).reshape(shape) * 0.1


with timer("attentionWeight_twice_matmul_type1"):
    r1 = attentionWeight_twice_matmul_type1(w, a)
    # print(r1)

with timer("attentionWeight_twice_matmul_type1_c"):
    r2 = attentionWeight_twice_matmul_type1_c(w, a)
    # print(r2)

with timer("attentionWeight_twice_matmul_type2"):
    r3 = attentionWeight_twice_matmul_type2(w, a)
    # print(r3)

with timer("attentionWeight_twice_matmul_type3"):
    r4 = attentionWeight_twice_matmul_type3(w, a)
    # print(r4)

with timer("attentionWeight_twice_matmul_type3_c"):
    r5 = attentionWeight_twice_matmul_type3_c(w, a)
    # print(r5)