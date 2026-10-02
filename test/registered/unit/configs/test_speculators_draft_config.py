"""Unit tests for vllm-speculators draft (DFlash/DSpark) config support.

These tests use synthetic on-disk ``config.json`` files instead of real
checkpoints, so they run on CPU-only CI.
"""

import json
from types import SimpleNamespace

import pytest

from sglang.srt.configs.model_config import ModelConfig, is_dspark_draft
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.utils.hf_transformers.config import (
    _try_load_speculators_draft_config,
)
from sglang.srt.utils.hf_transformers_utils import get_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _write_config(path, config_dict):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config_dict))


# A minimal vllm-speculators DSpark draft config: no top-level model_type,
# backbone config nested under ``transformer_layer_config``.
SPECULATORS_DRAFT_CONFIG = {
    "architectures": ["DSparkDraftModel"],
    "auto_map": {"AutoConfig": "speculators.dspark.modeling_dspark.DSparkDraftConfig"},
    "speculators_version": "0.0.1",
    "speculators_model_type": "dspark",
    "draft_vocab_size": 32000,
    "block_size": 7,
    "mask_token_id": 139,
    "markov_rank": 256,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "sample_from_anchor": True,
    "sliding_window_non_causal": False,
    "aux_hidden_state_layer_ids": [1, 10, 19, 28, 37],
    "dtype": "bfloat16",
    "transformer_layer_config": {
        "model_type": "qwen3",
        "vocab_size": 131072,
        "hidden_size": 3584,
        "intermediate_size": 9216,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 192,
        "max_position_embeddings": 262144,
        "sliding_window": 2048,
        "layer_types": ["sliding_attention"] * 5,
        "torch_dtype": "bfloat16",
    },
}


def test_speculators_draft_config_loads_without_model_type(tmp_path):
    draft_dir = tmp_path / "speculators_draft"
    _write_config(draft_dir, SPECULATORS_DRAFT_CONFIG)

    config = get_config(str(draft_dir), trust_remote_code=True)

    # The nested backbone config is built as its native HF config class.
    assert config.model_type == "qwen" or config.model_type == "qwen3"
    assert config.__class__.__name__ == "Qwen3Config"
    assert config.architectures == ["DSparkDraftModel"]

    # Backbone fields flattened from the nested config.
    assert config.vocab_size == 131072
    assert config.hidden_size == 3584
    assert config.num_hidden_layers == 5
    assert config.num_key_value_heads == 32

    # Serve-time draft fields lifted to the top level.
    assert config.draft_vocab_size == 32000
    assert config.block_size == 7
    assert config.mask_token_id == 139
    assert config.markov_rank == 256
    assert config.aux_hidden_state_layer_ids == [1, 10, 19, 28, 37]
    assert config.speculators_model_type == "dspark"
    assert str(config.torch_dtype) == "torch.bfloat16"


def test_regular_hf_config_unaffected(tmp_path):
    normal_dir = tmp_path / "normal_qwen3"
    _write_config(
        normal_dir,
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )

    config = get_config(str(normal_dir), trust_remote_code=True)
    assert config.__class__.__name__ == "Qwen3Config"
    assert config.architectures == ["Qwen3ForCausalLM"]
    # No speculators fields leaked onto a normal config.
    assert getattr(config, "draft_vocab_size", None) is None


def test_unknown_backbone_falls_through_to_autoconfig(tmp_path):
    bad_dir = tmp_path / "unknown_backbone"
    bad_config = dict(SPECULATORS_DRAFT_CONFIG)
    bad_config["transformer_layer_config"] = {
        "model_type": "definitely-not-a-real-backbone-xyz",
        "vocab_size": 8,
        "hidden_size": 4,
        "num_hidden_layers": 1,
    }
    _write_config(bad_dir, bad_config)

    # The parser contract: return None so default resolution takes over.
    assert _try_load_speculators_draft_config(str(bad_dir), None) is None

    # End-to-end, the config is still rejected (exact HF exception type varies
    # across transformers versions; trust_remote_code may attempt the auto_map
    # remote module first).
    with pytest.raises((ValueError, OSError)):
        get_config(str(bad_dir), trust_remote_code=True)


def test_aux_layer_ids_mapped_to_zero_indexed(tmp_path):
    draft_dir = tmp_path / "speculators_draft"
    _write_config(draft_dir, SPECULATORS_DRAFT_CONFIG)
    config = get_config(str(draft_dir), trust_remote_code=True)

    parsed = parse_dflash_draft_config(draft_hf_config=config)
    # speculators 1-indexed hidden-state ids [1, 10, 19, 28, 37] map to
    # 0-indexed decoder layer ids [0, 9, 18, 27, 36].
    assert parsed.target_layer_ids == [0, 9, 18, 27, 36]
    assert parsed.num_hidden_layers == 5

    dspark_cfg = parse_dspark_draft_config(draft_hf_config=config)
    assert dspark_cfg.resolve_gamma(default=None) == 7


@pytest.mark.parametrize(
    "aux_ids",
    [
        [],  # empty
        [1, 0],  # contains 0 -> negative 0-indexed id
        [-2, 10],  # directly negative after subtracting 1
    ],
)
def test_invalid_aux_layer_ids_rejected(aux_ids):
    fake_config = SimpleNamespace(
        num_hidden_layers=5,
        block_size=7,
        aux_hidden_state_layer_ids=aux_ids,
    )
    with pytest.raises(ValueError, match="aux_hidden_state_layer_ids"):
        parse_dflash_draft_config(draft_hf_config=fake_config)


def test_truncated_vocab_arch_routed_to_speculators_class(tmp_path):
    draft_dir = tmp_path / "speculators_draft"
    _write_config(draft_dir, SPECULATORS_DRAFT_CONFIG)

    model_config = ModelConfig(
        model_path=str(draft_dir),
        trust_remote_code=True,
        is_draft_model=True,
        speculative_algorithm="DSPARK",
    )
    assert model_config.hf_config.architectures == ["SpeculatorsDSparkDraftModel"]
    assert is_dspark_draft(model_config.hf_config)

    model_cls, arch = ModelRegistry.resolve_model_cls("SpeculatorsDSparkDraftModel")
    assert arch == "SpeculatorsDSparkDraftModel"
    assert model_cls.__name__ == "SpeculatorsDSparkDraftModel"
    assert model_cls.__module__ == "sglang.srt.models.dspark"


def test_full_vocab_draft_keeps_generic_arch(tmp_path):
    full_vocab_config = dict(SPECULATORS_DRAFT_CONFIG)
    full_vocab_config["draft_vocab_size"] = 131072
    draft_dir = tmp_path / "full_vocab_draft"
    _write_config(draft_dir, full_vocab_config)

    model_config = ModelConfig(
        model_path=str(draft_dir),
        trust_remote_code=True,
        is_draft_model=True,
        speculative_algorithm="DSPARK",
    )
    assert model_config.hf_config.architectures == ["DSparkDraftModel"]
    assert is_dspark_draft(model_config.hf_config)


def test_is_dspark_draft_arch_names():
    assert is_dspark_draft(SimpleNamespace(architectures=["DSparkDraftModel"]))
    assert is_dspark_draft(
        SimpleNamespace(architectures=["SpeculatorsDSparkDraftModel"])
    )
    assert not is_dspark_draft(SimpleNamespace(architectures=["Qwen3ForCausalLM"]))
