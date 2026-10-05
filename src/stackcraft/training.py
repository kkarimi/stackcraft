"""Optional native Clef training primitives; imported only by ML workflows."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as functional

from stackcraft.clef import ENCODING_VERSION, MODEL_ID, MODEL_REVISION, SOURCE_SHA256

FORMAT_VERSION = 1
HEAD_TYPE = "native-joint-schema-fp32-gathered-rows-v1"
_TARGET = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn|linear_attn|mlp)\."
    r"(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|"
    r"out_proj|gate_proj|up_proj|down_proj)$"
)


class GatheredFloat32Embedding:
    """The pinned head only indexes this object; never materialize all rows in FP32."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight

    def __getitem__(self, indices: torch.Tensor) -> torch.Tensor:
        # Both indexing and casting retain autograd edges when the input requires it.
        return self.weight[indices].float()


class FP32DecisionHead(nn.Module):
    """Keep the upstream head unchanged while adapting its floating inputs."""

    def __init__(self, native_head: nn.Module) -> None:
        super().__init__()
        self.native_head = native_head.float()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records: list[Any],
        output_embedding_weight: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        # Disable surrounding autocast so trainable head and its operations stay FP32.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return self.native_head(
                hidden_states.float(),
                input_ids,
                attention_mask,
                records,
                GatheredFloat32Embedding(output_embedding_weight),
            )


def lora_target_modules(backbone: nn.Module) -> list[str]:
    """Return full text-layer names; suffix-only matching can accidentally train vision."""
    targets = [
        name
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Linear) and _TARGET.fullmatch(name)
    ]
    if not targets:
        raise ValueError("no supported Qwen3.5 text-layer LoRA targets found")
    return sorted(targets)


def prepare_trainable(model: Any, mode: Literal["head", "lora"] = "head", rank: int = 4) -> Any:
    """Prepare a pinned, already admitted/loaded native ClefModel in place.

    This function loads no model or weights. The caller must supply the pinned
    native model, e.g. ClefPlayer.from_pretrained(...).model after memory admission.
    Backbone dtype is preserved; the intended release loader supplies BF16.
    """
    if mode not in ("head", "lora"):
        raise ValueError("training mode must be head or lora")
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if isinstance(model.head, FP32DecisionHead) or hasattr(model, "_stackcraft_training"):
        raise ValueError("model is already prepared for Stackcraft training")
    if hasattr(model.language_model, "peft_config"):
        raise ValueError("expected the unchanged backbone, not an existing PEFT model")
    model.language_model.requires_grad_(False)
    targets: list[str] = []
    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        targets = lora_target_modules(model.language_model)
        model.language_model = get_peft_model(
            model.language_model,
            LoraConfig(
                r=rank,
                lora_alpha=2 * rank,
                lora_dropout=0.0,
                target_modules=targets,
                bias="none",
            ),
        )
        if hasattr(model.language_model, "gradient_checkpointing_enable"):
            model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    model.head = FP32DecisionHead(model.head)
    model.head.requires_grad_(True)
    model.train()
    if mode == "head":
        model.language_model.eval()
    model._stackcraft_training = {
        "format_version": FORMAT_VERSION,
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "native_source_sha256": SOURCE_SHA256,
        "encoding_version": ENCODING_VERSION,
        "head_type": HEAD_TYPE,
        "mode": mode,
        "lora": (
            {"rank": rank, "alpha": 2 * rank, "dropout": 0.0, "target_modules": targets}
            if mode == "lora"
            else None
        ),
        "loss": {"label_smoothing": 0.05, "brier_weight": 0.1, "brier_reduction": "sum_options"},
    }
    return model


def decision_loss(
    logits: torch.Tensor,
    encoded_record: Any,
    target_action_id: str,
    *,
    label_smoothing: float = 0.05,
    brier_weight: float = 0.1,
) -> torch.Tensor:
    """One native choice: smoothed cross entropy plus multiclass Brier sum."""
    if len(encoded_record.questions) != 1:
        raise ValueError("training records must contain exactly one choice question")
    question = encoded_record.questions[0]
    if question.question_type != 1:
        raise ValueError("training question must be native choice type 1")
    ids = question.option_ids
    if tuple(ids) != tuple(sorted(set(ids))):
        raise ValueError("encoded option IDs must be unique and lexicographically sorted")
    if target_action_id not in ids:
        raise ValueError("target action is missing from native encoded option IDs")
    if logits.ndim != 1 or logits.numel() != len(ids):
        raise ValueError("logits shape does not match the encoded choices")
    if (
        not math.isfinite(label_smoothing)
        or not 0 <= label_smoothing <= 1
        or not math.isfinite(brier_weight)
        or brier_weight < 0
    ):
        raise ValueError("invalid label smoothing or Brier weight")
    if not torch.isfinite(logits).all():
        raise ValueError("decision logits contain nonfinite values")
    values = logits.float().unsqueeze(0)
    target = torch.tensor([ids.index(target_action_id)], device=values.device)
    cross_entropy = functional.cross_entropy(values, target, label_smoothing=label_smoothing)
    one_hot = functional.one_hot(target, num_classes=len(ids)).float()
    brier = (values.softmax(-1) - one_hot).square().sum(-1).mean()
    return cross_entropy + brier_weight * brier


def parameter_hashes(
    model: nn.Module, *, trainable: bool, chunk_elements: int = 1_048_576
) -> dict[str, str]:
    """Hash selected parameters exactly, moving only bounded chunks to CPU.

    Full frozen-backbone hashing is intentionally an explicit before/after audit,
    not a training-step operation. Dtype and shape are included in every digest.
    """
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("chunk_elements must be a positive integer")
    results = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != trainable:
            continue
        digest = hashlib.sha256()
        digest.update(f"{parameter.dtype}:{tuple(parameter.shape)}:".encode())
        flattened = parameter.detach().reshape(-1)
        for start in range(0, flattened.numel(), chunk_elements):
            chunk = flattened[start : start + chunk_elements].to("cpu").contiguous()
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
        results[name] = digest.hexdigest()
    return results


def save_checkpoint(
    model: Any, path: str | Path, *, extra_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Save head and optional LoRA separately, never the frozen multi-GB backbone."""
    if not isinstance(model.head, FP32DecisionHead) or not hasattr(model, "_stackcraft_training"):
        raise ValueError("model must be prepared before saving a training checkpoint")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    metadata = dict(model._stackcraft_training)
    metadata["extra"] = extra_metadata or {}
    # Store native head keys, not wrapper-specific state_dict prefixes.
    head_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.head.native_head.state_dict().items()
    }
    save_file(head_state, destination / "joint_head.safetensors")
    metadata["head_shapes"] = {name: list(tensor.shape) for name, tensor in head_state.items()}
    if metadata["mode"] == "lora":
        model.language_model.save_pretrained(destination / "adapter", safe_serialization=True)
    (destination / "training_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return metadata


def load_checkpoint(model: Any, path: str | Path, *, trainable: bool = False) -> Any:
    """Restore onto an unchanged pinned native base; reject incompatible metadata."""
    source = Path(path)
    metadata = json.loads((source / "training_config.json").read_text())
    expected = {
        "format_version": FORMAT_VERSION,
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "native_source_sha256": SOURCE_SHA256,
        "encoding_version": ENCODING_VERSION,
        "head_type": HEAD_TYPE,
    }
    for key, value in expected.items():
        if type(metadata.get(key)) is not type(value) or metadata[key] != value:
            raise ValueError(f"checkpoint {key} is incompatible with this pinned native adapter")
    mode = metadata.get("mode")
    if mode not in ("head", "lora"):
        raise ValueError("checkpoint training mode is invalid")
    if isinstance(model.head, FP32DecisionHead) or hasattr(model.language_model, "peft_config"):
        raise ValueError("checkpoint must load onto an unchanged native base")
    head_state = load_file(source / "joint_head.safetensors", device="cpu")
    shapes = {name: list(tensor.shape) for name, tensor in head_state.items()}
    base_shapes = {name: list(tensor.shape) for name, tensor in model.head.state_dict().items()}
    if shapes != metadata.get("head_shapes") or shapes != base_shapes:
        raise ValueError("checkpoint head structure differs from metadata or native model")
    if any(tensor.dtype != torch.float32 for tensor in head_state.values()):
        raise ValueError("checkpoint head tensors must be FP32")
    model.language_model.requires_grad_(False)
    if mode == "lora":
        from peft import PeftModel

        lora = metadata.get("lora")
        if not isinstance(lora, dict) or lora.get("target_modules") != lora_target_modules(
            model.language_model
        ):
            raise ValueError("checkpoint LoRA targets differ from the native text backbone")
        if (
            type(lora.get("rank")) is not int
            or lora["rank"] < 1
            or lora.get("alpha") != 2 * lora["rank"]
            or lora.get("dropout") != 0.0
        ):
            raise ValueError(
                "checkpoint LoRA rank, alpha or dropout violates the training contract"
            )
        config = json.loads((source / "adapter" / "adapter_config.json").read_text())
        if (
            config.get("r") != lora.get("rank")
            or config.get("lora_alpha") != lora.get("alpha")
            or config.get("lora_dropout") != lora.get("dropout")
            or sorted(config.get("target_modules", [])) != lora["target_modules"]
            or config.get("bias") != "none"
            or config.get("modules_to_save") is not None
        ):
            raise ValueError("saved adapter configuration differs from checkpoint metadata")
        model.language_model = PeftModel.from_pretrained(
            model.language_model, source / "adapter", is_trainable=trainable
        )
        if trainable and hasattr(model.language_model, "gradient_checkpointing_enable"):
            model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    elif metadata.get("lora") is not None:
        raise ValueError("head-only checkpoint must not contain LoRA configuration")
    model.head = FP32DecisionHead(model.head)
    model.head.native_head.load_state_dict(head_state, strict=True)
    model.head.requires_grad_(trainable)
    model._stackcraft_training = {
        key: value for key, value in metadata.items() if key not in ("extra", "head_shapes")
    }
    model.train(trainable)
    if mode == "head":
        model.language_model.eval()
    return model
