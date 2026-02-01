from oddkiva.brahma.torch.parallel.ddp import (
    get_rank
)


def format_msg(msg: str) -> str:
    rank = get_rank()
    if rank is not None:
        return f"[R:{rank}] {msg}"
    else:
        return msg
