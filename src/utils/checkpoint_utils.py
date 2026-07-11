import json
import logging
import os
import random
import re
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from utils.logging_utils import log_for_0, _process_index
from utils.train_utils import unwrap_model


def _local_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def upload_output_dir_to_hf(output_dir: str, hf_repo_id: Optional[str], reason: str = "artifacts"):
    if not hf_repo_id or _process_index() != 0:
        return
    folder_path = _local_path(output_dir)
    if not os.path.isdir(folder_path):
        log_for_0(f"HF upload skipped; output directory does not exist: {folder_path}",
                  level=logging.WARNING)
        return
    try:
        from huggingface_hub import HfApi
        repo_id = hf_repo_id.strip("/")
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        log_for_0(f"Uploading {reason} to HF: {repo_id}")
        api.upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="model")
        log_for_0(f"Uploaded {reason} to HF: {repo_id}")
    except Exception as e:
        log_for_0(f"Failed to upload {reason} to HF: {e}", level=logging.WARNING)


def _split_hf_path(path: str, min_parts: int) -> Optional[Tuple[str, str]]:
    if "://" in path:
        return None
    if path.startswith(("/", ".", "~")):
        return None
    if os.path.exists(_local_path(path)):
        return None
    parts = path.split("/")
    if len(parts) < min_parts:
        return None
    return "/".join(parts[:2]), "/".join(parts[2:])


def _capture_rng_state(state) -> Dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "dropout": (
            state.dropout_generator.get_state()
            if state.dropout_generator is not None else None
        ),
    }


def _gather_rng_states(state):
    local_state = _capture_rng_state(state)
    if not (dist.is_available() and dist.is_initialized()):
        return [local_state]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, local_state)
    return states


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_torch_save(payload: Dict[str, Any], path: str) -> int:
    fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        size = os.path.getsize(temp_path)
        os.replace(temp_path, path)
        _fsync_dir(os.path.dirname(path))
        return size
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _write_completion_marker(path: str, *, size: int, step: int) -> None:
    marker = f"{path}.complete"
    fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(marker)}.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"bytes": size, "step": int(step), "completed_at": time.time()}, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, marker)
        _fsync_dir(os.path.dirname(path))
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def save_checkpoint(state, output_dir: str, step: int, hf_repo_id: str = None):
    """Atomically save an optimizer-boundary checkpoint and completion marker."""
    if int(getattr(state, "accum_step", 0)) != 0:
        raise RuntimeError("refusing to checkpoint inside a gradient accumulation window")

    rng_states = _gather_rng_states(state)
    if _process_index() != 0:
        return
    ckpt_dir = _local_path(output_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    inner_model = unwrap_model(state.model)
    payload = {
        "params": inner_model.state_dict(),
        "ema_params1": state.ema_params1,
        "opt_state": state.optimizer.state_dict(),
        "lr_scheduler": state.lr_scheduler.state_dict() if state.lr_scheduler is not None else None,
        "step": int(state.step),
        "micro_step": int(getattr(state, "micro_step", state.step)),
        "optimizer_step": int(getattr(state, "optimizer_step", 0)),
        "accum_step": 0,
        "epoch": float(state.epoch),
        "rng_states": rng_states,
    }
    out_path = os.path.join(ckpt_dir, f"checkpoint_{step}")
    log_for_0(f"Saving checkpoint to {out_path}")
    size = _atomic_torch_save(payload, out_path)
    _write_completion_marker(out_path, size=size, step=step)
    log_for_0(f"Checkpoint committed to {out_path} ({size} bytes)")
    upload_output_dir_to_hf(output_dir, hf_repo_id, reason="checkpoint")


def _checkpoint_step(checkpoint_name: str) -> int:
    """Extract the trailing checkpoint step from a name; -1 if absent."""
    match = re.search(r"(\d+)$", checkpoint_name)
    return int(match.group(1)) if match else -1


def find_all_checkpoints(ckpt_dir: str, prefix: str = "checkpoint_"):
    """Find marker-validated local checkpoints, sorted by microstep ascending."""
    ckpt_dir = _local_path(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        return []
    pattern = re.compile(rf"^{re.escape(prefix)}\d+$")
    names = []
    for name in os.listdir(ckpt_dir):
        path = os.path.join(ckpt_dir, name)
        if not pattern.fullmatch(name) or not os.path.isfile(path):
            continue
        try:
            _validate_completion_marker(path)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        names.append(name)
    names.sort(key=_checkpoint_step)
    return [os.path.join(ckpt_dir, name) for name in names]


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "checkpoint_"):
    """Return the latest local checkpoint path, or None."""
    all_ckpts = find_all_checkpoints(ckpt_dir, prefix)
    return all_ckpts[-1] if all_ckpts else None


def _download_hf_checkpoint(checkpoint_path: str) -> Optional[str]:
    hf_path = _split_hf_path(checkpoint_path, min_parts=2)
    if hf_path is None:
        return None
    repo_id, sub_path = hf_path
    from huggingface_hub import snapshot_download
    log_for_0(f"Downloading checkpoint from HF: {repo_id}" + (f"/{sub_path}" if sub_path else ""))
    local_dir = snapshot_download(
        repo_id=repo_id, repo_type="model",
        allow_patterns=[f"{sub_path}/**"] if sub_path else None,
    )
    return os.path.join(local_dir, sub_path) if sub_path else local_dir


def _validate_completion_marker(path: str) -> None:
    marker_path = f"{path}.complete"
    if not os.path.isfile(marker_path):
        raise ValueError(f"checkpoint has no completion marker: {marker_path}")
    with open(marker_path, "r", encoding="utf-8") as handle:
        marker = json.load(handle)
    if not isinstance(marker, dict) or not isinstance(marker.get("bytes"), int):
        raise ValueError(f"checkpoint completion marker is invalid: {marker_path}")
    expected_step = _checkpoint_step(os.path.basename(path))
    if expected_step >= 0 and marker.get("step") != expected_step:
        raise ValueError(
            f"checkpoint step does not match completion marker: "
            f"expected {expected_step}, got {marker.get('step')!r}"
        )
    expected_size = marker["bytes"]
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise ValueError(
            f"checkpoint size does not match completion marker: expected {expected_size}, got {actual_size}"
        )


def _restore_checkpoint(checkpoint_path: str, *, require_complete: bool = False) -> Any:
    """Restore a checkpoint from a file or directory (latest inside dir)."""
    local = _local_path(checkpoint_path)
    if os.path.isdir(local):
        if require_complete:
            errors = []
            for candidate in reversed(find_all_checkpoints(local)):
                try:
                    _validate_completion_marker(candidate)
                    return torch.load(candidate, map_location="cpu")
                except Exception as exc:
                    errors.append(f"{os.path.basename(candidate)}: {exc}")
            detail = "; ".join(errors) if errors else "no completed checkpoints"
            raise ValueError(f"no valid completed checkpoint in {local}: {detail}")
        else:
            legacy = [
                os.path.join(local, name)
                for name in os.listdir(local)
                if re.fullmatch(r"checkpoint_\d+", name)
            ]
            resolved = max(legacy, key=lambda path: _checkpoint_step(os.path.basename(path)), default=None)
            if resolved is not None:
                return torch.load(resolved, map_location="cpu")
            return None
    if os.path.isfile(local):
        if require_complete:
            _validate_completion_marker(local)
        return torch.load(local, map_location="cpu")
    return None


def _validate_checkpoint(ckpt: Any, require_optimizer: bool = True):
    if ckpt is None:
        raise ValueError("checkpoint restore returned None")
    required_keys = ["params", "step", "epoch"]
    if require_optimizer:
        required_keys.append("opt_state")
    missing_keys = [key for key in required_keys if key not in ckpt]
    if missing_keys:
        raise ValueError(f"checkpoint restore missing keys: {missing_keys}")


def _load_checkpoint_payload(checkpoint_path: str, require_optimizer: bool) -> Tuple[Any, str]:
    """Load a checkpoint payload from local path or HF fallback."""
    ckpt, loaded_from = None, None
    errors = []

    local_path = _local_path(checkpoint_path)
    if os.path.exists(local_path):
        try:
            log_for_0(f"Loading local checkpoint from {local_path}...")
            # A directory is an implicit "latest" selection. Only committed
            # payloads may participate, including evaluation and warm starts.
            # Explicit files retain legacy/HF-compatible behavior for callers
            # that intentionally name one payload.
            require_complete = require_optimizer or os.path.isdir(local_path)
            ckpt = _restore_checkpoint(local_path, require_complete=require_complete)
            _validate_checkpoint(ckpt, require_optimizer=require_optimizer)
            loaded_from = "local"
        except Exception as e:
            errors.append(f"local: {e}")

    if ckpt is None:
        try:
            hf_path = _download_hf_checkpoint(checkpoint_path)
            if hf_path:
                log_for_0(f"Loading HF checkpoint from {hf_path}...")
                ckpt = _restore_checkpoint(hf_path, require_complete=False)
                _validate_checkpoint(ckpt, require_optimizer=require_optimizer)
                loaded_from = "HF"
        except Exception as e:
            errors.append(f"HF: {e}")
            log_for_0(f"HF checkpoint restore failed ({e}); falling back to local path.")

    if ckpt is None:
        raise ValueError(
            f"Failed to load checkpoint from {checkpoint_path}. Tried: {'; '.join(errors)}"
        )
    return ckpt, loaded_from


def load_checkpoint(checkpoint_path: str, state, load_optimizer: bool = True) -> Tuple[Any, int]:
    """Load an ELF checkpoint.

    Uses an existing local path first; otherwise tries HF and then local fallback.
    """
    log_for_0(f"Loading ELF checkpoint from {checkpoint_path}...")
    ckpt, loaded_from = _load_checkpoint_payload(
        checkpoint_path, require_optimizer=load_optimizer,
    )

    log_for_0(f"Loaded checkpoint keys: {list(ckpt.keys())}")

    inner_model = unwrap_model(state.model)
    inner_model.load_state_dict(ckpt["params"])
    ema_src = ckpt.get("ema_params1", ckpt["params"])
    device_map = {n: p.device for n, p in inner_model.named_parameters()}
    for n, b in inner_model.named_buffers():
        device_map.setdefault(n, b.device)
    fallback_device = next(iter(device_map.values()), torch.device("cpu"))
    state.ema_params1 = {
        n: t.to(device_map.get(n, fallback_device)) for n, t in ema_src.items()
    }
    if load_optimizer and state.optimizer is not None and ckpt.get("opt_state") is not None:
        state.optimizer.load_state_dict(ckpt["opt_state"])
    if load_optimizer and state.lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
        state.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    state.step = int(ckpt["step"])
    state.micro_step = int(ckpt.get("micro_step", ckpt["step"]))
    state.optimizer_step = int(ckpt.get("optimizer_step", state.micro_step))
    state.accum_step = int(ckpt.get("accum_step", 0))
    if state.accum_step != 0:
        raise ValueError("resume checkpoint was not saved at an optimizer boundary")
    state.epoch = float(ckpt["epoch"])
    rng_states = ckpt.get("rng_states")
    if rng_states:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        rng = rng_states[rank] if rank < len(rng_states) else rng_states[0]
        random.setstate(rng["python"])
        numpy_rng = rng["numpy"]
        np.random.set_state((
            numpy_rng["bit_generator"],
            np.asarray(numpy_rng["keys"], dtype=np.uint32),
            numpy_rng["position"],
            numpy_rng["has_gauss"],
            numpy_rng["cached_gaussian"],
        ))
        torch.set_rng_state(rng["torch_cpu"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        if rng.get("dropout") is not None and state.dropout_generator is not None:
            state.dropout_generator.set_state(rng["dropout"])
    elif ckpt.get("dropout_rng") is not None and state.dropout_generator is not None:
        # Legacy warm-start/resume compatibility for explicitly migrated files.
        state.dropout_generator.set_state(ckpt["dropout_rng"])

    step = int(ckpt["step"])
    log_for_0(f"Loaded {loaded_from} checkpoint from step {step} (epoch {state.epoch})")
    return state, step


def _init_ema_from_model(state) -> None:
    inner_model = unwrap_model(state.model)
    state.ema_params1 = {
        name: param.detach().clone()
        for name, param in inner_model.named_parameters()
    }


def load_warm_start_checkpoint(
    checkpoint_path: str,
    state,
    use_ema: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Partially initialize a model from a checkpoint without restoring training state.

    Only same-name, same-shape tensors are copied. Optimizer, scheduler, step,
    epoch, dropout RNG, and grad-accum buffers are intentionally left untouched.
    This is for trunk warm-starts such as old ELF -> sentence-plan ELF.
    """
    log_for_0(
        f"Warm-starting model from {checkpoint_path} "
        f"({'ema_params1' if use_ema else 'params'})..."
    )
    ckpt, loaded_from = _load_checkpoint_payload(checkpoint_path, require_optimizer=False)
    source_name = "ema_params1" if use_ema and ckpt.get("ema_params1") is not None else "params"
    source_state = ckpt[source_name]

    inner_model = unwrap_model(state.model)
    target_state = inner_model.state_dict()
    loadable = {}
    missing_keys = []
    shape_mismatch_keys = []
    unexpected_keys = []

    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            missing_keys.append(name)
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatch_keys.append(name)
            continue
        loadable[name] = source_tensor.to(dtype=target_tensor.dtype)

    for name in source_state.keys():
        if name not in target_state:
            unexpected_keys.append(name)

    if not loadable:
        raise ValueError(f"Warm-start from {checkpoint_path} found no matching tensors")

    merged_state = dict(target_state)
    merged_state.update(loadable)
    inner_model.load_state_dict(merged_state, strict=True)
    _init_ema_from_model(state)

    stats = {
        "source": source_name,
        "loaded_from": loaded_from,
        "checkpoint_step": int(ckpt.get("step", -1)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "loaded": len(loadable),
        "missing": len(missing_keys),
        "shape_mismatch": len(shape_mismatch_keys),
        "unexpected": len(unexpected_keys),
        "loaded_keys": sorted(loadable.keys()),
        "missing_keys": sorted(missing_keys),
        "shape_mismatch_keys": sorted(shape_mismatch_keys),
        "unexpected_keys": sorted(unexpected_keys),
    }
    log_for_0(
        "Warm-start loaded "
        f"{stats['loaded']} tensors from {loaded_from} checkpoint "
        f"(step={stats['checkpoint_step']}, epoch={stats['checkpoint_epoch']}); "
        f"missing={stats['missing']}, shape_mismatch={stats['shape_mismatch']}, "
        f"unexpected={stats['unexpected']}"
    )
    if missing_keys:
        log_for_0(f"Warm-start missing target keys (first 20): {stats['missing_keys'][:20]}")
    if shape_mismatch_keys:
        log_for_0(f"Warm-start shape-mismatch keys (first 20): {stats['shape_mismatch_keys'][:20]}")
    return state, stats
