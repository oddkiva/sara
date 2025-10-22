import os
import atexit

from torch.distributed import init_process_group, destroy_process_group


def ddp_setup():
    init_process_group(backend='nccl')

def get_local_rank():
    return int(os.environ['LOCAL_RANK'])

def get_rank():
    return int(os.environ['RANK'])

# --------------------------------------------------------------------------
# PARALLEL TRAINING
# Automatically clean up the parallel training environment.
# --------------------------------------------------------------------------
@atexit.register
def ddp_cleanup():
    local_rank = get_local_rank()
    rank = get_rank()
    print(f'[DDP][rank:{rank}][local_rank:{local_rank}] CLEANUP')
    destroy_process_group()
