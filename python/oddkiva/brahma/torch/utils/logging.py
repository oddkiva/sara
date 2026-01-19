from oddkiva.brahma.torch.parallel.ddp import (
    get_local_rank,
    get_rank
)


def format_msg(msg: str) -> str:
    local_rank = get_local_rank()
    rank = get_rank()
    if local_rank is not None and rank is not None:
        return f"[R:{rank},LR:{local_rank}] {msg}"
    else:
        return msg
