"""Public hand-shape benefit produced by a projected hero skill."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence


_RANKS = "3456789TJQKA2XD"
_RANK_INDEX = {rank: index for index, rank in enumerate(_RANKS)}


@dataclass(frozen=True)
class HandExpansionWeights:
    low_single_relief: float
    rank_upgrade: float
    structure_gain: float
    action_option_gain: float
    bomb_gain: float


HAND_EXPANSION_WEIGHTS = {
    "opening": HandExpansionWeights(3.0, 1.2, 2.4, 1.8, 3.0),
    "midgame": HandExpansionWeights(4.0, 1.5, 2.0, 1.5, 5.0),
    "endgame": HandExpansionWeights(5.0, 2.0, 2.5, 2.0, 5.0),
    "emergency": HandExpansionWeights(2.5, 2.5, 1.2, 1.0, 7.0),
}


@dataclass(frozen=True)
class HandExpansionUtility:
    low_single_relief: float
    rank_upgrade: float
    structure_gain: float
    action_option_gain: float
    bomb_gain: float
    expected_total: float
    worst_total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _normalise_rank(rank: str) -> str:
    return "X" if rank == "B" else "D" if rank == "R" else rank


def _remaining_hand(hand: Sequence[str], played: Sequence[str]) -> tuple[str, ...]:
    remaining = list(map(_normalise_rank, hand))
    for rank in map(_normalise_rank, played):
        if rank in remaining:
            remaining.remove(rank)
    return tuple(remaining)


def _low_single_burden(hand: Sequence[str]) -> float:
    counts = Counter(map(_normalise_rank, hand))
    ceiling = _RANK_INDEX["T"]
    straight_members: set[str] = set()
    current: list[str] = []
    for rank in _RANKS[:12]:
        if counts[rank] >= 1:
            current.append(rank)
        else:
            if len(current) >= 5:
                straight_members.update(current)
            current = []
    if len(current) >= 5:
        straight_members.update(current)
    return sum(
        (ceiling - _RANK_INDEX[rank] + 1) / (ceiling + 1)
        for rank, count in counts.items()
        if count == 1 and rank not in straight_members and rank in _RANK_INDEX and _RANK_INDEX[rank] <= ceiling
    )


def _average_rank(hand: Sequence[str]) -> float:
    values = [_RANK_INDEX.get(_normalise_rank(rank), _RANK_INDEX["2"]) for rank in hand]
    return sum(values) / len(values) if values else float(len(_RANKS))


def _group_value(count: int) -> float:
    if count >= 5:
        return 9.0 + 2.0 * (count - 5)
    if count == 4:
        return 6.0
    if count == 3:
        return 2.5
    if count == 2:
        return 1.0
    return 0.0


def _structure_value(hand: Sequence[str]) -> float:
    counts = Counter(map(_normalise_rank, hand))
    value = sum(_group_value(count) for count in counts.values())
    return value + (5.0 if counts["X"] and counts["D"] else 0.0)


def _run_options(counts: Counter[str], minimum_count: int, minimum_length: int) -> int:
    options = 0
    run_length = 0
    for rank in _RANKS[:12]:
        if counts[rank] >= minimum_count:
            run_length += 1
            if run_length >= minimum_length:
                options += run_length - minimum_length + 1
        else:
            run_length = 0
    return options


def _action_options(hand: Sequence[str]) -> int:
    counts = Counter(map(_normalise_rank, hand))
    options = len(counts)
    options += sum(count >= 2 for count in counts.values())
    options += sum(count >= 3 for count in counts.values())
    options += sum(count >= 4 for count in counts.values())
    options += sum(count >= 5 for count in counts.values())
    options += int(counts["X"] > 0 and counts["D"] > 0)
    options += _run_options(counts, 1, 5)
    options += _run_options(counts, 2, 3)
    options += _run_options(counts, 3, 2)
    return options


def _bomb_count(hand: Sequence[str]) -> int:
    counts = Counter(map(_normalise_rank, hand))
    return sum(count >= 4 for count in counts.values()) + int(counts["X"] > 0 and counts["D"] > 0)


def _branch_utility(
    baseline: Sequence[str],
    projected: Sequence[str],
    weights: HandExpansionWeights,
) -> tuple[float, ...]:
    low_relief = _low_single_burden(baseline) - _low_single_burden(projected)
    rank_upgrade = _average_rank(projected) - _average_rank(baseline)
    structure_gain = _structure_value(projected) - _structure_value(baseline)
    option_gain = float(_action_options(projected) - _action_options(baseline))
    bomb_gain = float(_bomb_count(projected) - _bomb_count(baseline))
    total = (
        weights.low_single_relief * low_relief
        + weights.rank_upgrade * rank_upgrade
        + weights.structure_gain * structure_gain
        + weights.action_option_gain * option_gain
        + weights.bomb_gain * bomb_gain
    )
    return low_relief, rank_upgrade, structure_gain, option_gain, bomb_gain, total


def evaluate_hand_expansion(
    hand: Sequence[str],
    physical_action: Sequence[str],
    projection: Any,
    game_stage: str = "midgame",
) -> HandExpansionUtility:
    """Compare the no-skill post-play hand with every projected skill branch."""

    baseline = _remaining_hand(hand, physical_action)
    weights = HAND_EXPANSION_WEIGHTS.get(game_stage, HAND_EXPANSION_WEIGHTS["midgame"])
    branches = tuple(getattr(projection, "random_branches", ()))
    if branches:
        outcomes = tuple((float(branch.probability), tuple(branch.hand)) for branch in branches)
    else:
        outcomes = ((1.0, tuple(getattr(projection, "post_hand", baseline))),)
    probability_sum = sum(max(0.0, probability) for probability, _ in outcomes) or 1.0
    values = [
        (max(0.0, probability) / probability_sum, _branch_utility(baseline, projected, weights))
        for probability, projected in outcomes
    ]
    expected = [sum(probability * utility[index] for probability, utility in values) for index in range(6)]
    return HandExpansionUtility(
        low_single_relief=round(expected[0], 6),
        rank_upgrade=round(expected[1], 6),
        structure_gain=round(expected[2], 6),
        action_option_gain=round(expected[3], 6),
        bomb_gain=round(expected[4], 6),
        expected_total=round(expected[5], 6),
        worst_total=round(min(utility[5] for _, utility in values), 6),
    )
