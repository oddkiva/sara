import os

from torch.distributed import init_process_group, destroy_process_group


def ddp_setup():
    init_process_group(backend='nccl')

def get_local_rank():
    return int(os.environ['LOCAL_RANK'])

def get_rank():
    return int(os.environ['RANK'])
