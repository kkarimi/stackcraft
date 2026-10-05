"""Tiny native-head training checks, not proof that the 9B GPU probe fits."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="training tests require the optional ML extra")
pytest.importorskip("peft", reason="training tests require the optional ML extra")

from stackcraft.clef import MODEL_ID, MODEL_REVISION, import_pinned_source  # noqa: E402
from stackcraft.training import (  # noqa: E402
    FP32DecisionHead,
    GatheredFloat32Embedding,
    decision_loss,
    load_checkpoint,
    lora_target_modules,
    parameter_hashes,
    prepare_trainable,
    save_checkpoint,
)


@pytest.fixture(scope="module")
def native():
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            MODEL_ID, "joint_schema_model.py", revision=MODEL_REVISION, local_files_only=True
        )
    except OSError:
        pytest.skip("pinned native source must be cached for actual-head CPU tests")
    return import_pinned_source(Path(path), trust_pinned_code=True)


def record(native):
    return native.EncodedRecord(
        input_ids=tuple(range(16)),
        questions=(
            native.EncodedQuestion(
                question_id="placement",
                question_type=1,
                question_span=(1, 3),
                option_spans=((4, 6), (7, 9), (10, 12)),
                option_ids=("r0x0", "r0x1", "r1x0"),
            ),
        ),
        record_id="tiny-test",
    )


class TinyText(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 8)
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(8, 8, bias=False)
        layer.linear_attn = torch.nn.Module()
        layer.linear_attn.in_proj_qkv = torch.nn.Linear(8, 8, bias=False)
        layer.linear_attn.in_proj_z = torch.nn.Linear(8, 8, bias=False)
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(8, 8, bias=False)
        self.layers = torch.nn.ModuleList([layer])

    def forward(self, input_ids, **kwargs):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = hidden + layer.mlp.down_proj(
                (
                    layer.self_attn.q_proj(hidden)
                    + layer.linear_attn.in_proj_qkv(hidden)
                    + layer.linear_attn.in_proj_z(hidden)
                ).tanh()
            )
        return SimpleNamespace(last_hidden_state=hidden)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {"model_type": "custom", "_name_or_path": "tiny-offline-test"}
        self.model = torch.nn.Module()
        self.model.language_model = TinyText()
        self.model.visual = torch.nn.Module()
        self.model.visual.q_proj = torch.nn.Linear(8, 8)
        self.mtp = torch.nn.Module()
        self.mtp.q_proj = torch.nn.Linear(8, 8)
        self.lm_head = torch.nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def tiny_model(native):
    torch.manual_seed(42)
    head = native.JointSchemaHead(
        hidden_size=8, width=8, routing_layers=1, layers=1, heads=2, feedforward=16
    )
    return native.ClefModel(TinyBackbone().to(torch.bfloat16), head.to(torch.bfloat16))


def test_gather_cast_matches_full_cast_and_preserves_selected_gradients():
    weight = torch.randn(32, 8, dtype=torch.bfloat16, requires_grad=True)
    selected = torch.tensor([2, 7, 2])
    output = GatheredFloat32Embedding(weight)[selected]
    assert torch.equal(output, weight.float()[selected])
    output.sum().backward()
    assert torch.all(weight.grad[2] == 2)
    assert torch.all(weight.grad[7] == 1)
    assert torch.count_nonzero(weight.grad).item() == 16


def test_actual_native_head_float32_wrapper_matches_reference_and_gradients(native):
    model = tiny_model(native)
    head = model.head.float().eval()
    wrapper = FP32DecisionHead(copy.deepcopy(head)).eval()
    encoded = record(native)
    batch = native.collate_records([encoded], 0, torch.device("cpu"))
    hidden = torch.randn(1, 16, 8, dtype=torch.bfloat16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    weight = torch.randn(32, 8, dtype=torch.bfloat16, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    wrapped = wrapper(hidden, batch["input_ids"], batch["attention_mask"], [encoded], weight)[0][0]
    reference = head(
        reference_hidden.float(),
        batch["input_ids"],
        batch["attention_mask"],
        [encoded],
        reference_weight.float(),
    )[0][0]
    assert wrapped.dtype == torch.float32
    torch.testing.assert_close(wrapped, reference, rtol=0, atol=0)
    wrapped.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0, atol=0)
    assert torch.count_nonzero(hidden.grad) > 0
    for (_, actual), (_, expected) in zip(
        wrapper.native_head.named_parameters(), head.named_parameters(), strict=True
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def test_lora_targets_use_full_names_and_exclude_vision_output_and_mtp(native):
    targets = lora_target_modules(tiny_model(native).language_model)
    assert targets == [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.self_attn.q_proj",
    ]


@pytest.mark.parametrize("mode", ["head", "lora"])
def test_real_native_forward_gradients_changed_intended_bytes_and_reload(native, tmp_path, mode):
    model = prepare_trainable(tiny_model(native), mode=mode)
    encoded = record(native)
    batch = native.collate_records([encoded], 0, torch.device("cpu"))
    assert isinstance(model.head, FP32DecisionHead)
    assert all(parameter.dtype == torch.float32 for parameter in model.head.parameters())
    assert all(
        parameter.dtype == torch.bfloat16
        for name, parameter in model.language_model.named_parameters()
        if "lora_" not in name
    )
    before = parameter_hashes(model, trainable=True, chunk_elements=7)
    frozen = parameter_hashes(model, trainable=False, chunk_elements=7)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.002)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)[0][0]
        loss = decision_loss(logits, encoded, "r0x1")
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.head.parameters() if p.grad is not None]
        assert all(torch.isfinite(grad).all() for grad in grads)
        assert any(torch.count_nonzero(grad) > 0 for grad in grads)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step()
    after = parameter_hashes(model, trainable=True, chunk_elements=7)
    assert any(before[name] != digest for name, digest in after.items() if name.startswith("head."))
    if mode == "lora":
        assert any(before[name] != digest for name, digest in after.items() if "lora_" in name)
    assert parameter_hashes(model, trainable=False, chunk_elements=7) == frozen
    model.eval()
    with torch.no_grad():
        expected = model(batch)[0][0].softmax(-1)
    destination = tmp_path / mode
    metadata = save_checkpoint(model, destination, extra_metadata={"scope": "tiny-cpu-plumbing"})
    assert metadata["mode"] == mode
    assert (destination / "joint_head.safetensors").exists()
    assert (destination / "adapter").exists() == (mode == "lora")
    if mode == "lora":
        adapter_config = json.loads((destination / "adapter" / "adapter_config.json").read_text())
        assert adapter_config["base_model_name_or_path"] == MODEL_ID
        assert adapter_config["revision"] == MODEL_REVISION
        assert adapter_config["target_modules"] == metadata["lora"]["target_modules"]
    restored = load_checkpoint(tiny_model(native), destination)
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    with torch.no_grad():
        actual = restored(batch)[0][0].softmax(-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_checkpoint(model, destination)


def test_loss_uses_sorted_choice_index_and_declared_formula(native):
    encoded = record(native)
    logits = torch.tensor([0.2, -0.4, 1.2], requires_grad=True)
    actual = decision_loss(logits, encoded, "r0x1")
    ce = torch.nn.functional.cross_entropy(
        logits.unsqueeze(0), torch.tensor([1]), label_smoothing=0.05
    )
    brier = ((logits.softmax(-1) - torch.tensor([0.0, 1.0, 0.0])) ** 2).sum()
    torch.testing.assert_close(actual, ce + 0.1 * brier)
    actual.backward()
    assert torch.count_nonzero(logits.grad) > 0
    with pytest.raises(ValueError, match="missing"):
        decision_loss(logits, encoded, "not-an-action")
    with pytest.raises(ValueError, match="nonfinite"):
        decision_loss(torch.tensor([0.0, float("nan"), 1.0]), encoded, "r0x1")


@pytest.mark.parametrize(
    "key,value",
    [
        ("format_version", True),
        ("format_version", 2),
        ("base_model", "other/model"),
        ("base_revision", "wrong"),
        ("native_source_sha256", "wrong"),
        ("encoding_version", "wrong"),
        ("head_type", "wrong"),
    ],
)
def test_checkpoint_rejects_wrong_contract(native, tmp_path, key, value):
    model = prepare_trainable(tiny_model(native))
    checkpoint = tmp_path / "checkpoint"
    metadata = save_checkpoint(model, checkpoint)
    metadata[key] = value
    (checkpoint / "training_config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=key):
        load_checkpoint(tiny_model(native), checkpoint)


def test_parameter_hashes_are_chunk_independent_and_track_dtype_and_values():
    model = torch.nn.Linear(4, 3).to(torch.bfloat16)
    first = parameter_hashes(model, trainable=True, chunk_elements=1)
    assert first == parameter_hashes(model, trainable=True, chunk_elements=10)
    with torch.no_grad():
        model.weight[0, 0] += 1
    second = parameter_hashes(model, trainable=True)
    assert first["weight"] != second["weight"]
    assert first["bias"] == second["bias"]
    assert first["bias"] != parameter_hashes(model.float(), trainable=True)["bias"]


def test_actual_tiny_qwen_hybrid_checkpointing_keeps_lora_gradients(native):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "max_position_embeddings": 64,
            "use_cache": False,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 1,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 8,
            "num_position_embeddings": 16,
        },
    )
    backbone = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    targets = lora_target_modules(backbone)
    assert len(targets) == 15  # 5 linear-attention + 4 full-attention + 6 MLP projections.
    assert sum("linear_attn" in name for name in targets) == 5
    assert all(name.startswith("model.language_model.layers.") for name in targets)
    head = tiny_model(native).head
    model = prepare_trainable(native.ClefModel(backbone, head), mode="lora")
    assert model.language_model.get_base_model().is_gradient_checkpointing
    encoded = record(native)
    batch = native.collate_records([encoded], 0, torch.device("cpu"))
    loss = decision_loss(model(batch)[0][0], encoded, "r0x1")
    assert torch.isfinite(loss)
    loss.backward()
    for part in ("linear_attn", "self_attn", "mlp"):
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for name, parameter in model.named_parameters()
            if "lora_" in name and part in name
        )


def test_peft_minimized_suffixes_reload_but_extra_targets_fail(native, tmp_path):
    def larger_tiny_model():
        model = tiny_model(native)
        text = model.language_model.model.language_model
        text.layers = torch.nn.ModuleList([copy.deepcopy(text.layers[0]) for _ in range(32)])
        return model

    model = prepare_trainable(larger_tiny_model(), mode="lora")
    compressed = sorted(model.language_model.peft_config["default"].target_modules)
    intended = model._stackcraft_training["lora"]["target_modules"]
    assert len(intended) == 128
    assert len(compressed) < len(intended)  # Exercise actual PEFT >=20 optimization.
    checkpoint = tmp_path / "compressed"
    save_checkpoint(model, checkpoint)
    config_path = checkpoint / "adapter" / "adapter_config.json"
    config = json.loads(config_path.read_text())
    assert config["target_modules"] == intended  # New saves are explicit and canonical.
    config["target_modules"] = compressed  # Recreate the immutable older probe format.
    config["base_model_name_or_path"] = "/old/local/cache/snapshot"
    config["revision"] = None
    config_path.write_text(json.dumps(config))
    restored = load_checkpoint(larger_tiny_model(), checkpoint)
    original_adapters = {
        name: parameter.detach() for name, parameter in model.named_parameters() if "lora_" in name
    }
    restored_adapters = {
        name: parameter.detach()
        for name, parameter in restored.named_parameters()
        if "lora_" in name
    }
    assert original_adapters.keys() == restored_adapters.keys()
    for name, expected in original_adapters.items():
        torch.testing.assert_close(restored_adapters[name], expected, rtol=0, atol=0)
    # Force an otherwise-valid suffix to also select the output layer.
    config["target_modules"] = compressed + ["lm_head"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="adapter configuration"):
        load_checkpoint(larger_tiny_model(), checkpoint)
