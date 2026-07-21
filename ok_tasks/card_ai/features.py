from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np

from ok_tasks.ai_model_adapter import CARD_ORDER
from ok_tasks.card_ai.heroes import HERO_REGISTRY
from ok_tasks.card_ai.rules import WILDCARD
from ok_tasks.card_ai.schema import POSITIONS


RANKS = tuple(CARD_ORDER) + (WILDCARD,)
HERO_NAMES = tuple(HERO_REGISTRY)
ACTION_TYPES = (
    "none",
    "solo",
    "pair",
    "trio",
    "trio_solo",
    "trio_pair",
    "straight",
    "pair_chain",
    "airplane",
    "bomb",
    "rocket",
    "unknown",
)
HASHED_SKILL_SIZE = 64
HASHED_SOURCE_SIZE = 16
HISTORY_LIMIT = 64
HISTORY_FEATURE_SIZE = len(RANKS) + len(POSITIONS) + len(ACTION_TYPES) + 1
STATIC_FEATURE_SIZE = len(RANKS) * 3 + len(POSITIONS) + 2 + len(HERO_NAMES) + HASHED_SKILL_SIZE + HASHED_SOURCE_SIZE
TEACHER_FEATURE_SIZE = len(RANKS) * 2


def _rank_counts(ranks: list[str] | tuple[str, ...]) -> list[float]:
    counts = Counter(ranks)
    return [float(counts[rank]) for rank in RANKS]


def _hash_bucket(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big") % size


def _hashed_mapping(values: dict[str, Any], size: int) -> list[float]:
    result = [0.0] * size
    for key, value in values.items():
        if isinstance(value, bool):
            numeric = float(value)
        elif isinstance(value, (int, float)):
            numeric = float(value)
        else:
            numeric = 1.0
        result[_hash_bucket(str(key), size)] += numeric
    return result


def encode_candidate(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, np.ndarray | int]:
    hand = list(observation.get("hand", []))
    hand_ranks = [card.get("rank", "") if isinstance(card, dict) else str(card) for card in hand]
    target_ranks = list(observation.get("target_ranks", []))
    action_ranks = list(action.get("ranks", action.get("cards", [])))
    observer = observation.get("observer", observation.get("position", "landlord_down"))
    role = [float(observer == position) for position in POSITIONS]
    opponents = list(observation.get("opponent_card_counts", [17, 17]))[:2]
    opponents += [17] * (2 - len(opponents))
    opponent_counts = [float(value) / 30.0 for value in opponents]
    hero = observation.get("hero")
    hero_vector = [float(hero == name) for name in HERO_NAMES]
    hero_state = observation.get("hero_state", {}) if isinstance(observation.get("hero_state"), dict) else {}
    skill_values = {
        **(hero_state.get("skill_uses", {}) if isinstance(hero_state.get("skill_uses"), dict) else {}),
        **(hero_state.get("marks", {}) if isinstance(hero_state.get("marks"), dict) else {}),
    }
    sources = Counter()
    for card in hand:
        if not isinstance(card, dict):
            continue
        sources[str(card.get("source", "unknown"))] += 1
        for tag in card.get("tags", []):
            sources[f"tag:{tag}"] += 1
    static = np.asarray(
        _rank_counts(hand_ranks)
        + _rank_counts(target_ranks)
        + _rank_counts(action_ranks)
        + role
        + opponent_counts
        + hero_vector
        + _hashed_mapping(skill_values, HASHED_SKILL_SIZE)
        + _hashed_mapping(dict(sources), HASHED_SOURCE_SIZE),
        dtype=np.float32,
    )
    if static.shape != (STATIC_FEATURE_SIZE,):
        raise ValueError(f"静态特征应为 {STATIC_FEATURE_SIZE} 维，实际 {static.shape}")
    history = np.zeros((HISTORY_LIMIT, HISTORY_FEATURE_SIZE), dtype=np.float32)
    history_mask = np.zeros(HISTORY_LIMIT, dtype=np.bool_)
    raw_history = list(observation.get("history", []))[-HISTORY_LIMIT:]
    for index, event in enumerate(raw_history):
        ranks = list(event.get("ranks", []))
        actor = event.get("actor")
        kind = event.get("action_type", "none")
        kind = kind if kind in ACTION_TYPES else "unknown"
        row = (
            _rank_counts(ranks)
            + [float(actor == position) for position in POSITIONS]
            + [float(kind == name) for name in ACTION_TYPES]
            + [float(event.get("kind") == "pass")]
        )
        history[index] = np.asarray(row, dtype=np.float32)
        history_mask[index] = True
    return {
        "static": static,
        "history": history,
        "history_mask": history_mask,
        "role_index": POSITIONS.index(observer) if observer in POSITIONS else 2,
    }


def encode_teacher_hidden(full_state: dict[str, Any], observer: str) -> np.ndarray:
    players = full_state.get("players", {}) if isinstance(full_state, dict) else {}
    opponents = [position for position in POSITIONS if position != observer]
    values = []
    for position in opponents:
        hand = players.get(position, {}).get("hand", [])
        values.extend(_rank_counts([card.get("rank", "") for card in hand if isinstance(card, dict)]))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (TEACHER_FEATURE_SIZE,):
        return np.zeros(TEACHER_FEATURE_SIZE, dtype=np.float32)
    return result


def opponent_targets(full_state: dict[str, Any], observer: str) -> np.ndarray:
    return encode_teacher_hidden(full_state, observer).reshape(2, len(RANKS))


def teammate_target(full_state: dict[str, Any], observer: str) -> float:
    if observer == "landlord":
        return 0.0
    teammate = next(position for position in POSITIONS if position not in {"landlord", observer})
    return float(int(full_state.get("players", {}).get(teammate, {}).get("hand_count", 17)) <= 2)
