from collections import defaultdict
import psutil
from tabulate import tabulate
import pickle

from src.data.dataloader import DataLoader, default_collate_fn
from src.data import transforms as T
from src.data.coco.coco_dataset import CocoDetection, CocoDetection_memory_shared
from multiprocessing import Manager

from rtest.utils import *
from rtest.utils import *
import torch.utils.data as data

"""
testing memory usage of dataloader.

requires psutil and tabulate
"""

def test_dataloader(
        dataset_class,
        worker_init_fn,
        range_num,
        img_folder="./dataset/coco/train2017/",
        ann_file="./dataset/coco/annotations/instances_train2017.json",
        batch_size=4,
        shuffle=True, 
        num_workers=4):

    train_dataset = dataset_class(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms = T.Compose([T.RandomPhotometricDistort(p=0.5), 
                                T.RandomZoomOut(fill=0), 
                                T.RandomIoUCrop(p=0.8),
                                T.SanitizeBoundingBox(min_size=1),
                                T.RandomHorizontalFlip(),
                                T.Resize(size=[640, 640]),
                                # transforms.Resize(size=639, max_size=640),
                                # # transforms.PadToSize(spatial_size=640),
                                T.ToImageTensor(),
                                T.ConvertDtype(),
                                T.SanitizeBoundingBox(min_size=1),
                                T.ConvertBox(out_fmt='cxcywh', normalize=True)]),
        return_masks=False,
        remap_mscoco_category=True)
    
    train_dataset = torch.utils.data.Subset(train_dataset, range(range_num))
    
    return DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        collate_fn=default_collate_fn, 
        drop_last=True, 
        worker_init_fn=worker_init_fn)


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


def main(dataset_class):
    def hook_pid(worker_id):
        pid = os.getpid()
        pids.append(pid)
        print(f"tracking {worker_id} PID: {pid}")

    manager = Manager()
    pids = manager.list()
    pids.append(os.getpid())
    dataloader = test_dataloader(dataset_class=dataset_class, worker_init_fn=hook_pid, range_num=10000, batch_size=32, num_workers=4)

    t = time.time()
    for i, (samples, targets) in enumerate(dataloader):
        samples = pickle.dumps(samples)
        targets = pickle.dumps(targets)

        if i % 10 == 0:
            
            datas = {pid: get_mem_info(pid) for pid in pids}

            table = []
            keys = list(list(datas.values())[0].keys())
            now = str(int(time.perf_counter() % 1e5))

            for pid, data in datas.items():
                table.append((now, str(pid)) + tuple(format(data[k]) for k in keys))

            print(tabulate(table, headers=["time", "PID"] + keys))
            print(f"totle pss : {sum([k[1]['pss'] / 1024 / 1024 / 1024 for k in datas.items()]):.3f}GB")
            print(f"iteration : {i} / {len(dataloader)}, time : {time.time() - t:.3f}")
            t = time.time()

if __name__ == '__main__':
	main(CocoDetection)
	main(CocoDetection_memory_shared)