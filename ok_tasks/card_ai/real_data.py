from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ok_tasks.card_ai.inference import legacy_state_to_observation
from ok_tasks.card_ai.schema import POSITIONS, TrajectoryEvent
from ok_tasks.card_ai.trajectory import TrajectoryWriter


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            values.append(json.loads(line))
        except ValueError:
            return []
    return values


def _team_rewards(position: str, won: bool) -> dict[str, float]:
    user_reward = 1.0 if won else -1.0
    if position == "landlord":
        return {"landlord": user_reward, "landlord_down": -user_reward, "landlord_up": -user_reward}
    teammate = next(value for value in POSITIONS if value not in {"landlord", position})
    return {"landlord": -user_reward, position: user_reward, teammate: user_reward}


def convert_real_runs(runs_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root)
    converted = 0
    decisions = 0
    skipped = []
    for game_folder in sorted(Path(runs_root).glob("*/games/game_*")):
        summary_path = game_folder / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except ValueError:
            skipped.append({"game": str(game_folder), "reason": "summary_json"})
            continue
        if summary.get("status") != "completed" or summary.get("won") is None:
            continue
        events = _read_jsonl(game_folder / "events.jsonl")
        states = {event.get("round_id"): event for event in events if event.get("event_type") == "decision_state"}
        game_events = []
        for decision in [event for event in events if event.get("event_type") == "decision"]:
            state = states.get(decision.get("round_id"))
            if state is None or not decision.get("candidates"):
                continue
            position = state.get("position", "landlord_down")
            observation = legacy_state_to_observation(state)
            game_events.append(
                TrajectoryEvent(
                    game_id=f"real_{game_folder.parent.parent.name}_{game_folder.name}",
                    sequence=len(game_events) + 1,
                    event_type="decision",
                    actor=position,
                    observation=observation,
                    legal_actions=tuple(dict(value) for value in decision.get("candidates", [])),
                    chosen_action={"ranks": list(decision.get("chosen", [])), "kind": "play"},
                    metadata={"source": "real", "original_path": str(game_folder)},
                )
            )
        if not game_events:
            continue
        position = game_events[-1].actor or "landlord_down"
        rewards = _team_rewards(position, bool(summary.get("won")))
        terminal = TrajectoryEvent(
            game_id=game_events[-1].game_id,
            sequence=len(game_events) + 1,
            event_type="terminal",
            rewards=rewards,
            terminal=True,
            metadata={"source": "real"},
        )
        path = output / f"{game_events[-1].game_id}.jsonl.gz"
        TrajectoryWriter(path).extend([*game_events, terminal])
        converted += 1
        decisions += len(game_events)
    return {"converted_games": converted, "decisions": decisions, "skipped": skipped[:100]}
