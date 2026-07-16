from packaging.version import parse as parse_version
import pytest
import torch
import torch.nn as nn

from modules.t5_encoder import T5Encoder, prepare_t5_attention_mask


def test_new_transformers_3d_keep_mask_becomes_4d_additive_mask():
    keep = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.float32)

    mask = prepare_t5_attention_mask(
        keep,
        expand_3d=True,
        model_dtype=torch.float32,
    )

    assert mask.shape == (1, 1, 2, 2)
    assert mask[0, 0, 0, 0] == 0
    assert mask[0, 0, 1, 1] == 0
    assert mask[0, 0, 0, 1] == torch.finfo(torch.float32).min


def test_legacy_transformers_keeps_3d_boolean_mask_for_internal_conversion():
    keep = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.float32)

    mask = prepare_t5_attention_mask(
        keep,
        expand_3d=False,
        model_dtype=torch.float32,
    )

    assert mask.dtype == torch.bool
    assert mask.shape == (1, 2, 2)
    assert torch.equal(mask, keep.bool())


def test_new_transformers_t5_prefix_is_invariant_to_future_tokens():
    transformers = pytest.importorskip("transformers")
    if parse_version(transformers.__version__) < parse_version("4.45.0"):
        pytest.skip("new 4D prepared-mask API is unavailable")
    from transformers import T5Config, T5EncoderModel

    config = T5Config(
        vocab_size=64,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
    )
    wrapper = T5Encoder.__new__(T5Encoder)
    nn.Module.__init__(wrapper)
    wrapper.model = T5EncoderModel(config).eval()
    wrapper._expand_3d_attention_mask = True

    first = torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]])
    changed_future = torch.tensor([[10, 11, 12, 13, 30, 31, 32, 33]])
    keep = torch.tensor(
        [[
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        first_hidden = wrapper(first, keep, deterministic=True)
        changed_hidden = wrapper(changed_future, keep, deterministic=True)

    assert torch.equal(first_hidden[:, :4], changed_hidden[:, :4])
    assert not torch.equal(first_hidden[:, 4:], changed_hidden[:, 4:])
