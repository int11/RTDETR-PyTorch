"""
Copyright (c) 2025 int11. All Rights Reserved.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
from collections import defaultdict
import psutil
from tabulate import tabulate
import pickle
import torch
import time
import torch
import multiprocessing as mp

from multiprocessing import Manager
from src.zoo import *
from src.data.coco import CocoDetection, CocoDetection_share_memory
from src.data.dataloader import DataLoader, BatchImageCollateFuncion

"""
testing memory usage of dataloader.

requires psutil and tabulate
"""
class MemoryMonitor():
    def __init__(self, pids: list[int] = None):
        if pids is None:
            pids = [os.getpid()]
        self.pids = pids

    def add_pid(self, pid: int):
        assert pid not in self.pids
        self.pids.append(pid)

    def _refresh(self):
        self.data = {pid: self.get_mem_info(pid) for pid in self.pids}
        return self.data

    def table(self) -> str:
        self._refresh()
        table = []
        keys = list(list(self.data.values())[0].keys())
        now = str(int(time.perf_counter() % 1e5))
        for pid, data in self.data.items():
            table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
        return tabulate(table, headers=["time", "PID"] + keys)

    def str(self):
        self._refresh()
        keys = list(list(self.data.values())[0].keys())
        res = []
        for pid in self.pids:
            s = f"PID={pid}"
            for k in keys:
                v = self.format(self.data[pid][k])
                s += f", {k}={v}"
            res.append(s)
        return "\n".join(res)

    @staticmethod
    def format(size: int) -> str:
        for unit in ('', 'K', 'M', 'G'):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)
    
    @staticmethod
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


def main(args):
    def hook_pid(worker_id):
        pid = os.getpid()
        monitor.pids.append(pid)
        print(f"tracking {worker_id} PID: {pid}")

    monitor = MemoryMonitor()
    monitor.pids = Manager().list(monitor.pids)

    dataset_class = CocoDetection_share_memory if args.dataset_class == 'CocoDetection_share_memory' else CocoDetection
    
    dataset = coco_train_dataset(
        img_folder=args.img_folder,
        ann_file=args.ann_file,
        range_num=args.range_num,
        dataset_class=dataset_class
    )
    dataloader = DataLoader(
        dataset=dataset, 
        worker_init_fn=hook_pid,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=BatchImageCollateFuncion())

    t = time.time()

    for i, (samples, targets) in enumerate(dataloader):
        # fake read the data
        samples = pickle.dumps(samples)
        targets = pickle.dumps(targets)

        if i % 10 == 0:
            print(monitor.table())
            print(f"totle pss : {sum([k[1]['pss'] / 1024 / 1024 / 1024 for k in monitor.data.items()]):.3f}GB")
            print(f"iteration : {i} / {len(dataloader)}, time : {time.time() - t:.3f}")
            t = time.time()

def main2(args):
    def worker(_, dataset: torch.utils.data.Dataset):
        while True:
            for sample in dataset:
                result = pickle.dumps(sample)

    start_method = 'fork'
    mp.set_start_method(start_method)
    monitor = MemoryMonitor()
    
    dataset_class = CocoDetection_share_memory if args.dataset_class == 'CocoDetection_share_memory' else CocoDetection
    
    ds = coco_train_dataset(
        img_folder=args.img_folder,
        ann_file=args.ann_file,
        range_num=args.range_num,
        dataset_class=dataset_class
    )
    print(monitor.table())
    if start_method == "forkserver":
        # Reduce 150M-per-process USS due to "import torch".
        mp.set_forkserver_preload(["torch"])

    ctx = torch.multiprocessing.start_processes(
        worker, (ds, ), nprocs=4, join=False,
        daemon=True, start_method=start_method)
    [monitor.add_pid(pid) for pid in ctx.pids()]

    try:
        while True:
            print(monitor.table())
            print(f"totle pss : {sum([k[1]['pss'] / 1024 / 1024 / 1024 for k in monitor.data.items()]):.3f}GB")
            time.sleep(1)
    finally:
        ctx.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Memory Check Tool', add_help=False)
    
    # Dataset options
    parser.add_argument('--dataset_class', type=str, default='CocoDetection_share_memory',
                        choices=['CocoDetection', 'CocoDetection_share_memory'],
                        help='Dataset class to use')
    parser.add_argument('--share_memory', action='store_true', default=True,
                        help='Enable shared memory for dataset')
    parser.add_argument('--range_num', type=int, default=30000,
                        help='Number of samples to use from dataset')
    
    # Dataset paths
    parser.add_argument('--img_folder', type=str, default='datasets/coco/train2017',
                        help='Path to image folder')
    parser.add_argument('--ann_file', type=str, default='datasets/coco/annotations/instances_train2017.json',
                        help='Path to annotation file')
    
    # Dataloader options
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for dataloader')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for dataloader')
    args = parser.parse_args()

    main(args)