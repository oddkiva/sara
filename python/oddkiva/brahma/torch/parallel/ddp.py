import atexit

from torch.distributed import init_process_group, destroy_process_group


def ddp_setup():
    init_process_group(backend='nccl')


# --------------------------------------------------------------------------
# PARALLEL TRAINING
# Automatically clean up the parallel training environment.
# --------------------------------------------------------------------------
@atexit.register
def ddp_cleanup():
    destroy_process_group()
