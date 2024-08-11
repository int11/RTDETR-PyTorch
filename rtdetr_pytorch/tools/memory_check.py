from collections import defaultdict
import psutil
from tabulate import tabulate
import pickle
import torch
from src.data.coco.coco_dataset import CocoDetection
from src.data.dataloader import DataLoader, default_collate_fn
from src.data import transforms as T
import torchvision
from multiprocessing import Manager

from rtest.utils import *
from rtest.utils import *
import torch.utils.data as data


def get_mem_info(pid: int) -> dict[str, int]:
  res = defaultdict(int)
  for mmap in psutil.Process(pid).memory_maps():
    res['rss'] += mmap.rss
    res['pss'] += mmap.pss
    res['uss'] += mmap.private_clean + mmap.private_dirty
    res['shared'] += mmap.shared_clean + mmap.shared_dirty
    if mmap.path.startswith('/'):
      res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
  return res

def format(size: int) -> str:
    for unit in ('', 'K', 'M', 'G'):
      if size < 1024:
        break
      size /= 1024.0
    return "%.1f%s" % (size, unit)


def rtdetr_train_dataloader1(
        img_folder="./dataset/coco/train2017/",
        ann_file="./dataset/coco/annotations/instances_train2017.json", 
        range_num=None,
        batch_size=4,
        shuffle=True, 
        num_workers=4):

    train_dataset = torchvision.datasets.CocoDetection(
        root=img_folder,
        annFile=ann_file,
        transforms = T.Compose([T.Resize(size=[640, 640]),
                                # transforms.Resize(size=639, max_size=640),
                                # # transforms.PadToSize(spatial_size=640),
                                T.ToImageTensor(),
                                T.ConvertDtype(),
                                T.SanitizeBoundingBox(min_size=1),
                                T.ConvertBox(out_fmt='cxcywh', normalize=True)]))
    
    if range_num != None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(range_num))

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=default_collate_fn, drop_last=True, worker_init_fn=worker_init_fn)


manager = Manager()
pids = manager.list()

def worker_init_fn(worker_id):
    pid = os.getpid()
    pids.append(pid)
    print(f"Worker {worker_id} PID: {pid}")


if __name__ == '__main__':
    pids.append(os.getpid()) 
    train_dataloader = rtdetr_train_dataloader1(batch_size=32, num_workers=0)

    for samples, targets in train_dataloader:
        samples = pickle.dumps(samples)
        targets = pickle.dumps(targets)
        data = {pid: get_mem_info(pid) for pid in pids} 

        table = []
        keys = list(list(data.values())[0].keys())
        now = str(int(time.perf_counter() % 1e5))

        for pid, data in data.items():
            table.append((now, str(pid)) + tuple(format(data[k]) for k in keys))

        print(tabulate(table, headers=["time", "PID"] + keys))