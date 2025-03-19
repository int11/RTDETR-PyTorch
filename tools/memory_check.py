"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

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
from rtest.utils import *
from src.data.coco import CocoDetection, CocoDetection_share_memory

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


def main(**kwargs):
    def hook_pid(worker_id):
        pid = os.getpid()
        monitor.pids.append(pid)
        print(f"tracking {worker_id} PID: {pid}")

    monitor = MemoryMonitor()
    monitor.pids = Manager().list(monitor.pids)

    dataset = coco_train_dataset(**kwargs)
    dataloader = DataLoader(
        dataset=dataset, 
        worker_init_fn=hook_pid,
        batch_size=32, 
        num_workers=2,
        shuffle=False)

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

def main2(**kwargs):
    def worker(_, dataset: torch.utils.data.Dataset):
        while True:
            for sample in dataset:
                result = pickle.dumps(sample)

    start_method = 'fork'
    mp.set_start_method(start_method)
    monitor = MemoryMonitor()
    ds = coco_train_dataset(**kwargs)
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
    # main(dataset_class=CocoDetection, range_num=30000)
    # main(dataset_class=CocoDetection_share_memory, share_memory=False, range_num=30000)
    main(dataset_class=CocoDetection_share_memory, share_memory=True, range_num=30000)