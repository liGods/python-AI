"""Named decision stages derived only from public card counts."""

from __future__ import annotations

from enum import StrEnum

from .context import DecisionContext


class DecisionStage(StrEnum):
    OPENING = "opening"
    MIDGAME = "midgame"
    ENDGAME = "endgame"
    EMERGENCY = "emergency"


def classify_stage(context: DecisionContext) -> DecisionStage:
    if context.urgent:
        return DecisionStage.EMERGENCY
    if len(context.hand) <= 10:
        return DecisionStage.ENDGAME
    if context.nearest_enemy <= 10:
        return DecisionStage.MIDGAME
    return DecisionStage.OPENING
