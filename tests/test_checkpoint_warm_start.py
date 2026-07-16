import torch
import torch.nn as nn

from utils.checkpoint_utils import load_warm_start_checkpoint
from utils.train_utils import TrainState


class WarmStartModel(nn.Module):
    def __init__(self, changed_in_features=2, with_extra=False):
        super().__init__()
        self.shared = nn.Linear(2, 2, bias=False)
        self.changed = nn.Linear(changed_in_features, 2, bias=False)
        if with_extra:
            self.extra = nn.Linear(2, 1, bias=False)


def make_state(model):
    return TrainState(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        ema_params1=TrainState.init_ema(model),
        step=0,
        epoch=0,
        dropout_generator=torch.Generator(device="cpu").manual_seed(0),
    )


def test_warm_start_loads_matching_tensors_and_skips_new_or_mismatched(tmp_path):
    old_model = WarmStartModel(changed_in_features=2)
    with torch.no_grad():
        old_model.shared.weight.fill_(2.0)
        old_model.changed.weight.fill_(3.0)
    ckpt_path = tmp_path / "checkpoint_10"
    torch.save(
        {
            "params": old_model.state_dict(),
            "step": 10,
            "epoch": 2,
        },
        ckpt_path,
    )

    new_model = WarmStartModel(changed_in_features=3, with_extra=True)
    with torch.no_grad():
        new_model.shared.weight.zero_()
        new_model.changed.weight.zero_()
        new_model.extra.weight.fill_(5.0)
    state = make_state(new_model)

    state, stats = load_warm_start_checkpoint(str(ckpt_path), state)

    assert stats["loaded"] == 1
    assert stats["loaded_keys"] == ["shared.weight"]
    assert stats["shape_mismatch_keys"] == ["changed.weight"]
    assert stats["missing_keys"] == ["extra.weight"]
    assert state.step == 0
    assert state.epoch == 0
    assert torch.allclose(state.model.shared.weight, torch.full_like(state.model.shared.weight, 2.0))
    assert torch.allclose(state.model.changed.weight, torch.zeros_like(state.model.changed.weight))
    assert torch.allclose(state.model.extra.weight, torch.full_like(state.model.extra.weight, 5.0))
    assert torch.allclose(state.ema_params1["shared.weight"], state.model.shared.weight)


def test_warm_start_can_load_from_ema_params(tmp_path):
    model = WarmStartModel()
    params = model.state_dict()
    ema_params = {name: torch.full_like(tensor, 7.0) for name, tensor in params.items()}
    ckpt_path = tmp_path / "checkpoint_3"
    torch.save(
        {
            "params": params,
            "ema_params1": ema_params,
            "step": 3,
            "epoch": 1,
        },
        ckpt_path,
    )

    target = WarmStartModel()
    state, stats = load_warm_start_checkpoint(str(ckpt_path), make_state(target), use_ema=True)

    assert stats["source"] == "ema_params1"
    assert stats["loaded"] == len(target.state_dict())
    assert torch.allclose(state.model.shared.weight, torch.full_like(state.model.shared.weight, 7.0))
