"""Offline adapter-contract tests, not evidence of real Clef inference quality."""

import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from stackcraft.clef import (
    ENCODING_VERSION,
    MODEL_ID,
    QUESTION_ID,
    ClefPlayer,
    complete_token_count,
    encode_observation,
    import_pinned_source,
    observation_record,
)
from stackcraft.engine import new_game
from stackcraft.players import observe


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[ord(character) for character in text])


def fake_native():
    """Fake only the boundary plumbing; never stands in for an ML acceptance run."""
    native = SimpleNamespace(SYSTEM_PROMPT="test system prompt")

    def encode(tokenizer, record, *, max_length):
        count = complete_token_count(tokenizer, native, record)
        assert count <= max_length
        return SimpleNamespace(
            input_ids=tuple(range(count)),
            questions=(
                SimpleNamespace(
                    question_id=QUESTION_ID,
                    option_ids=tuple(sorted(record["questions"][QUESTION_ID]["criteria"])),
                ),
            ),
        )

    native.encode_record = Mock(side_effect=encode)
    return native


def test_module_import_does_not_load_any_ml_dependency():
    script = """
import sys
import stackcraft.clef
assert not {'torch', 'transformers', 'huggingface_hub'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_native_record_contains_all_actions_and_only_visible_state():
    observation = observe(new_game(42))
    record = observation_record(observation)
    assert record["model"] == MODEL_ID
    assert record["state"]["encoding_version"] == ENCODING_VERSION
    assert record["state"]["board_rows"] == ["." * 10] * 20
    assert record["state"]["current_piece"] == observation.current
    assert record["state"]["next_piece"] == observation.next_piece
    assert "seed" not in json.dumps(record)
    question = record["questions"][QUESTION_ID]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == [action.id for action in observation.legal_actions]
    for action in observation.legal_actions:
        description = question["criteria"][action.id]
        assert description == {
            "rotation": action.rotation,
            "column": action.x,
            "landing_row": action.y,
            "cells": [list(cell) for cell in action.cells],
        }


def test_hidden_seed_and_non_gameplay_colors_cannot_change_record():
    state = new_game(42)
    assert observation_record(observe(state)) == observation_record(observe(replace(state, seed=9)))
    board = list(state.board)
    board[-1] = (1,) + (0,) * 9
    occupied = replace(state, board=tuple(board))
    other_board = list(board)
    other_board[-1] = (7,) + (0,) * 9
    assert observation_record(observe(occupied)) == observation_record(
        observe(replace(occupied, board=tuple(other_board)))
    )


def test_empty_or_duplicate_choices_and_wrong_rules_fail():
    observation = observe(new_game(42))
    for changed in (
        replace(observation, legal_actions=()),
        replace(observation, legal_actions=observation.legal_actions * 2),
        replace(observation, rules_version="v2"),
    ):
        with pytest.raises(ValueError):
            observation_record(changed)


def test_context_preflight_rejects_before_native_encoder_can_truncate():
    observation = observe(new_game(42))
    native = fake_native()
    tokenizer = CharacterTokenizer()
    count = complete_token_count(tokenizer, native, observation_record(observation))
    with pytest.raises(ValueError, match="refusing to truncate"):
        encode_observation(observation, tokenizer, native, count - 1)
    native.encode_record.assert_not_called()
    encoded = encode_observation(observation, tokenizer, native, count)
    assert len(encoded.input_ids) == count
    assert encoded.questions[0].option_ids == tuple(sorted(a.id for a in observation.legal_actions))


def test_count_tokenizes_native_segments_separately():
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=[1]))
    record = observation_record(observe(new_game(42)))
    native = fake_native()
    count = complete_token_count(tokenizer, native, record)
    options = len(record["questions"][QUESTION_ID]["criteria"])
    assert count == tokenizer.call_count == 8 + 3 * options
    segments = [call.args[0] for call in tokenizer.call_args_list]
    assert segments[0] == "\n\nSCHEMA FIELDS:\n"
    assert segments[-1] == json.dumps(record["state"], separators=(",", ":"), sort_keys=True)


def test_encoder_drift_is_not_silently_accepted():
    native = fake_native()
    native.encode_record.side_effect = None
    native.encode_record.return_value = SimpleNamespace(input_ids=(1,))
    with pytest.raises(ValueError, match="source contract changed"):
        encode_observation(observe(new_game(42)), CharacterTokenizer(), native, 100_000)


def test_wrong_option_mapping_fails_even_with_right_length():
    observation = observe(new_game(42))
    native = fake_native()
    original = native.encode_record.side_effect

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result.questions[0].option_ids = result.questions[0].option_ids[::-1]
        return result

    native.encode_record.side_effect = changed
    with pytest.raises(ValueError, match="option IDs"):
        encode_observation(observation, CharacterTokenizer(), native, 100_000)


def test_untrusted_or_changed_source_is_not_executed(tmp_path: Path):
    source = tmp_path / "joint_schema_model.py"
    source.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="trust_pinned_code"):
        import_pinned_source(source)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        import_pinned_source(source, trust_pinned_code=True)
    with pytest.raises(ValueError, match="trust_pinned_code"):
        ClefPlayer.from_pretrained()


@pytest.mark.parametrize("tie", [False, True])
def test_probability_mapping_uses_encoded_ids_and_engine_tiebreak(monkeypatch, tie):
    observation = observe(new_game(42))
    observation = replace(observation, legal_actions=observation.legal_actions[::-1])
    native = fake_native()
    native.collate_records = Mock(return_value={"batch": "test"})
    count = len(observation.legal_actions)
    values = (
        [1 / count] * count
        if tie
        else [index / sum(range(1, count + 1)) for index in range(1, count + 1)]
    )
    logits = Mock()
    logits.float.return_value.softmax.return_value.tolist.return_value = values
    model = Mock(return_value=[[logits]])
    model.eval.return_value = model
    model.parameters.return_value = iter([SimpleNamespace(device="fake", dtype="fake-float")])
    context = Mock(__enter__=Mock(), __exit__=Mock(return_value=False))
    torch = SimpleNamespace(inference_mode=Mock(return_value=context))
    real_import = importlib.import_module
    monkeypatch.setattr(
        "stackcraft.clef.importlib.import_module",
        lambda name: torch if name == "torch" else real_import(name),
    )
    player = ClefPlayer(
        model,
        SimpleNamespace(tokenizer=CharacterTokenizer()),
        native,
        revision="test-fake-only",
        max_length=100_000,
    )
    model.parameters.assert_not_called()
    assert player.runtime_config["dtype"] is None
    assert player.runtime_config["device"] is None
    assert player.runtime_config["model_id"] == MODEL_ID
    assert player.runtime_config["revision"] == "test-fake-only"
    assert player.runtime_config["encoding_version"] == ENCODING_VERSION
    assert player.runtime_config["max_length"] == 100_000
    decision = player.choose(observation)
    expected = (
        observation.legal_actions[0].id
        if tie
        else sorted(action.id for action in observation.legal_actions)[-1]
    )
    assert decision.action_id == expected
    assert decision.probabilities == dict(
        zip(sorted(a.id for a in observation.legal_actions), values, strict=True)
    )
    assert player.last_input_tokens is not None and player.last_input_tokens > 0
    assert player.runtime_config["dtype"] == "fake-float"
    assert player.runtime_config["device"] == "fake"
    model.parameters.assert_called_once()
    logits.float.return_value.softmax.assert_called_once_with(-1)
    model.assert_called_once_with({"batch": "test"})


@pytest.mark.skipif(
    os.environ.get("STACKCRAFT_TEST_NATIVE_ENCODING") != "1",
    reason="opt-in native CPU encoding check needs ML dependencies and pinned cached tokenizer",
)
def test_real_pinned_tokenizer_and_native_encoder_agree_without_model_load():
    from stackcraft.clef import MODEL_REVISION
    from stackcraft.schema import PIECES

    hub = importlib.import_module("huggingface_hub")
    transformers = importlib.import_module("transformers")
    path = Path(
        hub.hf_hub_download(
            MODEL_ID,
            "joint_schema_model.py",
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    ).parent
    native = import_pinned_source(path / "joint_schema_model.py", trust_pinned_code=True)
    processor = transformers.AutoProcessor.from_pretrained(path, local_files_only=True)
    measured_counts = []
    for dense in (False, True):
        for piece in PIECES:
            state = replace(new_game(42), current=piece)
            if dense:
                board = tuple([(0,) * 10] * 5) + tuple(
                    tuple(1 if (x + y) % 3 else 0 for x in range(10)) for y in range(15)
                )
                state = replace(state, board=board)
            observation = observe(state)
            count = complete_token_count(
                processor.tokenizer, native, observation_record(observation)
            )
            encoded = encode_observation(observation, processor.tokenizer, native, 4096)
            assert len(encoded.input_ids) == count
            assert len(encoded.questions[0].option_ids) == len(observation.legal_actions)
            with pytest.raises(ValueError, match="refusing to truncate"):
                encode_observation(observation, processor.tokenizer, native, count - 1)
            measured_counts.append(count)
    # Golden counts bind both our record text and the pinned tokenizer/encoder.
    assert measured_counts == [
        1296,
        848,
        2248,
        1296,
        1296,
        2248,
        2248,
        1286,
        878,
        2153,
        1286,
        1286,
        2153,
        2153,
    ]
