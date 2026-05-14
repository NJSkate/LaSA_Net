import functools
import os
import os.path as osp
import pickle
import shutil
import tempfile

import torch
from torch import distributed as dist

def is_main_process():
    rank, _ = get_dist_info()
    return rank == 0

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def init_dist(backend='nccl', **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper

def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part

    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')

    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)

    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]

    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))

        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        ordered_results = ordered_results[:size]
        return ordered_results

def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()

    if tmpdir is None:
        MAX_LEN = 512

        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            os.makedirs('.dist_test', exist_ok=True)
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        os.makedirs(tmpdir, exist_ok=True)

    pickle.dump(result_part, open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb'))
    dist.barrier()

    if rank != 0:
        return None
    else:

        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(pickle.load(open(part_file, 'rb')))

        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        ordered_results = ordered_results[:size]

        shutil.rmtree(tmpdir)
        return ordered_results
