"""Pinned native Clef integration; importing this module needs no ML packages."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from stackcraft.players import Decision, Observation, validate_decision
from stackcraft.schema import RULES_VERSION

MODEL_ID = "Cloudflare/clef-flash"
MODEL_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
SOURCE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
ENCODING_VERSION = "stackcraft-clef-v1"
QUESTION_ID = "placement"
DEFAULT_MAX_LENGTH = 4096

_INSTRUCTIONS = (
    "Choose the legal placement that maximizes total lines cleared over the game. "
    "Avoid holes and high stacks so future pieces can be placed. Only the current "
    "piece and exactly one next piece are known. Consider all supplied placements."
)


def observation_record(observation: Observation) -> dict[str, Any]:
    """Build one native choice question from visible state; never include a seed."""
    if observation.rules_version != RULES_VERSION:
        raise ValueError("unsupported observation rules version")
    actions = observation.legal_actions
    if not actions:
        raise ValueError("cannot ask Clef to choose with no legal placements")
    if len({action.id for action in actions}) != len(actions):
        raise ValueError("legal placement IDs must be unique")
    return {
        "model": MODEL_ID,
        "state": {
            "encoding_version": ENCODING_VERSION,
            "rules_version": RULES_VERSION,
            "board_rows": [
                "".join("#" if cell else "." for cell in row) for row in observation.board
            ],
            "current_piece": observation.current,
            "next_piece": observation.next_piece,
            "coordinates": (
                "10 columns x=0..9 left to right; 20 rows y=0..19 top to bottom. "
                "board_rows are top to bottom; . is empty and # is occupied. "
                "Placement cells use absolute [x,y] coordinates."
            ),
            "rules": (
                "Place the current four-cell piece at one supplied legal landing. "
                "A vertical hard drop starts fully inside row 0; no tucks, wall kicks, "
                "hold or gravity timer. Full rows clear simultaneously; rows above fall. "
                "Score for 1/2/3/4 cleared rows is 100/300/500/800. "
                "The next piece becomes current. No legal placement means game over. "
                "Future pieces beyond the one preview are unknown."
            ),
        },
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": {
                    action.id: {
                        "rotation": action.rotation,
                        "column": action.x,
                        "landing_row": action.y,
                        "cells": [list(cell) for cell in action.cells],
                    }
                    for action in actions
                },
            }
        },
    }


def _render(value: Any) -> str:
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def complete_token_count(tokenizer: Any, native: Any, record: dict[str, Any]) -> int:
    """Count exact text segments used by the pinned native encoder, before encoding.

    Tokenizing the joined text is NOT equivalent: native encode_record tokenizes
    each segment separately. This implementation is coupled to SOURCE_SHA256.
    Only the one-question, text-only Stackcraft record schema is supported.
    """
    if record.get("images") or record.get("videos"):
        raise ValueError("Stackcraft Clef encoding is text-only")
    questions = record.get("questions", {})
    if list(questions) != [QUESTION_ID] or questions[QUESTION_ID].get("type") != "choice":
        raise ValueError("expected the single Stackcraft placement choice question")
    question = questions[QUESTION_ID]
    segments = [
        "\n\nSCHEMA FIELDS:\n",
        f"\nFIELD 1\nID: {QUESTION_ID}\nTYPE: choice\nINSTRUCTION: ",
        _render(question["instructions"]),
        "\nALLOWED OPTIONS:\n",
    ]
    for index, (option_id, description) in enumerate(sorted(question["criteria"].items())):
        semantics = {"option_id": option_id}
        if description is not None:
            semantics["description"] = description
        segments.extend((f"OPTION {index + 1}: ", _render(semantics), "\n"))
    segments.extend(
        (
            "END FIELD\n",
            f"<|im_start|>system\n{native.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
            _render(record["state"]),
        )
    )
    return sum(len(tokenizer(segment, add_special_tokens=False).input_ids) for segment in segments)


def encode_observation(
    observation: Observation, tokenizer: Any, native: Any, max_length: int = DEFAULT_MAX_LENGTH
) -> Any:
    """Reject overlong input before native encode_record can truncate the board."""
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    record = observation_record(observation)
    required = complete_token_count(tokenizer, native, record)
    if required > max_length:
        raise ValueError(
            f"complete Stackcraft state and choices require {required} tokens; "
            f"max_length={max_length}; refusing to truncate"
        )
    encoded = native.encode_record(tokenizer, record, max_length=max_length)
    if len(encoded.input_ids) != required:
        raise ValueError("native encoding length differs from preflight; source contract changed")
    expected_ids = tuple(sorted(action.id for action in observation.legal_actions))
    if (
        len(encoded.questions) != 1
        or encoded.questions[0].question_id != QUESTION_ID
        or encoded.questions[0].option_ids != expected_ids
    ):
        raise ValueError("native encoded option IDs differ from complete legal action set")
    return encoded


def import_pinned_source(path: Path, *, trust_pinned_code: bool = False) -> ModuleType:
    """Execute only the explicitly trusted, reviewed native source bytes.

    A pinned revision limits changes; it does not make Python code a sandbox.
    The caller must deliberately accept executing the reviewed upstream module.
    """
    if not trust_pinned_code:
        raise ValueError("native Clef loading requires trust_pinned_code=True")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("native Clef source SHA256 mismatch; refusing to execute")
    name = f"_stackcraft_clef_native_{SOURCE_SHA256}"
    if name in sys.modules:
        return sys.modules[name]
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module  # dataclasses resolves annotations through this registry.
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


class ClefPlayer:
    """Native single-record inference with unrounded probabilities.

    Direct construction also supports an explicitly supplied trained native model;
    such callers must set revision to the actual checkpoint identity. The factory
    below loads only the pinned, unchanged upstream release.
    """

    name = "clef-flash"

    def __init__(
        self,
        model: Any,
        processor: Any,
        native: Any,
        *,
        revision: str,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if type(max_length) is not int or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        self.model = model.eval()
        self.processor = processor
        self.native = native
        self.revision = revision
        self.max_length = max_length
        self.last_input_tokens: int | None = None
        self.runtime_config: dict[str, Any] = {
            "model_id": MODEL_ID,
            "revision": revision,
            "base_revision": MODEL_REVISION,
            "encoding_version": ENCODING_VERSION,
            "source_sha256": SOURCE_SHA256,
            "max_length": max_length,
            "dtype": None,
            "device": None,
        }

    @classmethod
    def from_pretrained(
        cls,
        *,
        trust_pinned_code: bool = False,
        local_files_only: bool = True,
        device: str = "cuda",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> ClefPlayer:
        """Load a pinned local snapshot; downloading is separately opt-in.

        This loads real 9B weights onto device. It performs no GPU admission or
        workload management; callers must establish available memory beforehand.
        """
        if not trust_pinned_code:
            raise ValueError("native Clef loading requires trust_pinned_code=True")
        if type(max_length) is not int or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        hub = importlib.import_module("huggingface_hub")
        snapshot = Path(
            hub.snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=local_files_only,
            )
        )
        native = import_pinned_source(
            snapshot / "joint_schema_model.py", trust_pinned_code=trust_pinned_code
        )
        config = json.loads((snapshot / "config.json").read_text())
        context = config["text_config"]["max_position_embeddings"]
        if max_length > context:
            raise ValueError(f"max_length={max_length} exceeds backbone context={context}")
        torch = importlib.import_module("torch")
        model, processor = native.load_release_model(snapshot, device=device, dtype=torch.bfloat16)
        return cls(
            model,
            processor,
            native,
            revision=f"{MODEL_ID}@{MODEL_REVISION}:{ENCODING_VERSION}",
            max_length=max_length,
        )

    def choose(self, observation: Observation) -> Decision:
        tokenizer = self.processor.tokenizer
        encoded = encode_observation(observation, tokenizer, self.native, self.max_length)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("native Clef tokenizer has no padding token")
        torch = importlib.import_module("torch")
        first_parameter = next(self.model.parameters())
        device = first_parameter.device
        batch = self.native.collate_records([encoded], pad_id, device)
        with torch.inference_mode():
            result = self.model(batch)
            if len(result) != 1 or len(result[0]) != 1:
                raise ValueError("native Clef must return one batch and one question")
            logits = result[0][0]
            values = logits.float().softmax(-1).tolist()
        ids = encoded.questions[0].option_ids
        if len(values) != len(ids):
            raise ValueError("native Clef probability count differs from legal choices")
        probabilities = dict(zip(ids, values, strict=True))
        # Preserve common engine ordering on exact ties, not native lexical order.
        selected = max(observation.legal_actions, key=lambda action: probabilities[action.id])
        decision = Decision(selected.id, probabilities)
        validate_decision(decision, observation)
        self.last_input_tokens = len(encoded.input_ids)
        self.runtime_config.update(
            dtype=str(first_parameter.dtype),
            device=str(device),
        )
        return decision
