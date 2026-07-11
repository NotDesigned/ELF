import inspect
import json
import logging
import os


def _process_index() -> int:
    """Return torch.distributed rank, falling back to env vars or 0."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def log_for_0(msg, *args, level=logging.INFO):
    """Log only on the first process (rank == 0)."""
    if _process_index() != 0:
        return
    caller_module = inspect.currentframe().f_back.f_globals.get("__name__", __name__)
    logging.getLogger(caller_module).log(level, msg, *args)


def append_jsonl_for_0(path, record) -> bool:
    """Append one structured record on rank 0 and make it visible immediately."""
    if _process_index() != 0:
        return False

    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return True
