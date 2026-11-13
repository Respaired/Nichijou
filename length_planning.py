"""CPU-only worker entry point for startup length-group planning."""
import os


os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import multiprocessing as mp
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from dataset import _init_length_plan_worker, _plan_length_partition


def main():
    
    directory, workers = sys.argv[1], int(sys.argv[2])
    tasks = np.load(os.path.join(directory, 'tasks.npy'), mmap_mode='r')
    
    with open(os.path.join(directory, 'tokenizer.pkl'), 'rb') as source:
        tokenizer, options = pickle.load(source)
        
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
            initializer=_init_length_plan_worker,
            initargs=(directory, tokenizer, options)) as pool:
        
        
        futures = {}
        for worker in range(workers):
            selected = tasks[tasks[:, 2] == worker, :2]
            if len(selected):
                futures[pool.submit(_plan_length_partition, selected)] = (worker, len(selected))
                
                
        checked, completed = 0, 0
        for future in as_completed(futures):
            worker, size = futures[future]
            result = future.result()
            np.savez(os.path.join(directory, f'result_{worker}.npz'),
                     **dict(zip(('rows', 'offsets', 'frames', 'durations', 'ranks'), result)))
            checked += size
            completed += 1
            
            
            print(f'Length grouping: {checked:,}/{len(tasks):,} starts completed '
                  f'({completed}/{len(futures)} worker partitions)', flush=True)


if __name__ == '__main__':
    main()
