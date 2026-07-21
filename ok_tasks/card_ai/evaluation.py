from __future__ import annotations

import math
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable

from ok_tasks.card_ai.heroes import HERO_REGISTRY, OWNED_HEROES
from ok_tasks.card_ai.policies import LegacyStableRulePolicy, Policy, StableRulePolicy
from ok_tasks.card_ai.schema import POSITIONS
from ok_tasks.card_ai.self_play import SelfPlayRunner
from ok_tasks.card_ai.trajectory import atomic_json


def _reward_for(position: str, winner: str) -> float:
    landlord_won = winner == "landlord"
    return 1.0 if (position == "landlord") == landlord_won else -1.0


def wilson_interval(successes: int, samples: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial success rate."""

    if samples < 0 or successes < 0 or successes > samples:
        raise ValueError("Wilson 区间要求 0 <= successes <= samples")
    if samples == 0:
        return 0.0, 1.0
    rate = successes / samples
    denominator = 1.0 + z * z / samples
    center = (rate + z * z / (2.0 * samples)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / samples + z * z / (4.0 * samples * samples)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate_hero_policy_paired(
    heroes: tuple[str, ...] | None = None,
    deals_per_hero: int = 3,
    seed: int = 20260718,
    maximum_steps: int = 1000,
) -> dict[str, Any]:
    """Compare the shared skill policy with the frozen generic policy on identical deals."""

    selected_heroes = tuple(heroes or HERO_REGISTRY)
    unknown = tuple(hero for hero in selected_heroes if hero not in HERO_REGISTRY)
    if unknown:
        raise ValueError(f"未注册武将: {', '.join(unknown)}")
    if deals_per_hero <= 0:
        raise ValueError("deals_per_hero must be positive")

    hero_reports: dict[str, dict[str, Any]] = {}
    all_failures: list[dict[str, Any]] = []
    for hero_index, hero in enumerate(selected_heroes):
        candidate_wins = 0
        baseline_wins = 0
        paired_improvements = 0
        paired_regressions = 0
        paired_ties = 0
        completed = 0
        failures: list[dict[str, Any]] = []
        for deal_index in range(deals_per_hero):
            deal_seed = seed + hero_index * deals_per_hero + deal_index
            for position in POSITIONS:
                hero_map = {seat: hero if seat == position else None for seat in POSITIONS}
                candidate_policies: dict[str, Policy] = {
                    seat: LegacyStableRulePolicy() for seat in POSITIONS
                }
                candidate_policies[position] = StableRulePolicy()
                baseline_policies: dict[str, Policy] = {
                    seat: LegacyStableRulePolicy() for seat in POSITIONS
                }
                try:
                    _, candidate_summary = SelfPlayRunner(candidate_policies).run_game(
                        deal_seed,
                        hero_map,
                        maximum_steps,
                        include_full_state=False,
                    )
                    _, baseline_summary = SelfPlayRunner(baseline_policies).run_game(
                        deal_seed,
                        hero_map,
                        maximum_steps,
                        include_full_state=False,
                    )
                except Exception as error:
                    failure = {
                        "hero": hero,
                        "seed": deal_seed,
                        "position": position,
                        "error": str(error),
                    }
                    failures.append(failure)
                    all_failures.append(failure)
                    continue

                candidate_reward = _reward_for(position, candidate_summary["winner"])
                baseline_reward = _reward_for(position, baseline_summary["winner"])
                candidate_wins += int(candidate_reward > 0)
                baseline_wins += int(baseline_reward > 0)
                paired_improvements += int(candidate_reward > baseline_reward)
                paired_regressions += int(candidate_reward < baseline_reward)
                paired_ties += int(candidate_reward == baseline_reward)
                completed += 1

        requested = deals_per_hero * len(POSITIONS)
        candidate_interval = wilson_interval(candidate_wins, completed)
        baseline_interval = wilson_interval(baseline_wins, completed)
        passed = (
            completed == requested
            and not failures
            and candidate_interval[0] + 1e-12 >= baseline_interval[0]
        )
        hero_reports[hero] = {
            "requested_samples": requested,
            "completed_samples": completed,
            "candidate_wins": candidate_wins,
            "candidate_win_rate": candidate_wins / completed if completed else 0.0,
            "candidate_wilson_95": list(candidate_interval),
            "baseline_wins": baseline_wins,
            "baseline_win_rate": baseline_wins / completed if completed else 0.0,
            "baseline_wilson_95": list(baseline_interval),
            "paired_improvements": paired_improvements,
            "paired_regressions": paired_regressions,
            "paired_ties": paired_ties,
            "legal_action_rate": 1.0 if completed == requested and not failures else completed / requested,
            "state_errors": len(failures),
            "sim_verified": passed,
            "failures": failures,
        }

    return {
        "rules_version": "3p.1",
        "seed": seed,
        "deals_per_hero": deals_per_hero,
        "heroes": hero_reports,
        "requested_heroes": len(selected_heroes),
        "sim_verified_heroes": sum(report["sim_verified"] for report in hero_reports.values()),
        "passed": bool(hero_reports) and all(report["sim_verified"] for report in hero_reports.values()),
        "failed_samples": len(all_failures),
        "failures": all_failures[:100],
    }


def evaluate_paired(
    candidate: Policy,
    stable: Policy | None = None,
    deals: int = 50_000,
    seed: int = 20260718,
    maximum_steps: int = 1000,
    deal_offset: int = 0,
) -> dict[str, Any]:
    baseline = stable or StableRulePolicy()
    deltas = []
    candidate_wins = 0
    completed = 0
    failures = []
    for deal_index in range(deals):
        deal_seed = seed + deal_index
        global_deal_index = deal_offset + deal_index
        hero_map = {
            position: OWNED_HEROES[(global_deal_index * len(POSITIONS) + index) % len(OWNED_HEROES)]
            for index, position in enumerate(POSITIONS)
        }
        for position in POSITIONS:
            candidate_policies = {seat: baseline for seat in POSITIONS}
            candidate_policies[position] = candidate
            try:
                _, candidate_summary = SelfPlayRunner(candidate_policies).run_game(
                    deal_seed, hero_map, maximum_steps, include_full_state=False
                )
                _, stable_summary = SelfPlayRunner({seat: baseline for seat in POSITIONS}).run_game(
                    deal_seed, hero_map, maximum_steps, include_full_state=False
                )
            except Exception as error:
                failures.append({"seed": deal_seed, "position": position, "error": str(error)})
                continue
            candidate_reward = _reward_for(position, candidate_summary["winner"])
            stable_reward = _reward_for(position, stable_summary["winner"])
            deltas.append(candidate_reward - stable_reward)
            candidate_wins += int(candidate_reward > 0)
            completed += 1
    mean_delta = fmean(deltas) if deltas else 0.0
    standard_error = stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else float("inf")
    margin = 1.96 * standard_error
    win_rate = candidate_wins / completed if completed else 0.0
    bounded = min(1.0 - 1e-6, max(1e-6, win_rate))
    return {
        "requested_deals": deals,
        "paired_seat_samples": completed,
        "mean_reward_delta": mean_delta,
        "confidence_lower": mean_delta - margin,
        "confidence_upper": mean_delta + margin,
        "candidate_win_rate": win_rate,
        "elo_estimate": 400.0 * math.log10(bounded / (1.0 - bounded)),
        "illegal_actions": 0,
        "failed_samples": len(failures),
        "failures": failures[:100],
        "_delta_sum": float(sum(deltas)),
        "_delta_square_sum": float(sum(delta * delta for delta in deltas)),
        "_candidate_wins": candidate_wins,
    }


_WORKER_CANDIDATE: Policy | None = None
_WORKER_BASELINE: Policy | None = None


def _initialize_openvino_worker(model_path: str, policy_id: str) -> None:
    global _WORKER_CANDIDATE, _WORKER_BASELINE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENVINO_CPU_THREADS_NUM", "1")
    from ok_tasks.card_ai.policies import AcceleratedPolicy

    backend = os.environ.get("CARD_AI_INFERENCE_BACKEND", "auto")
    _WORKER_CANDIDATE = AcceleratedPolicy(model_path, policy_id, backend=backend)
    _WORKER_BASELINE = StableRulePolicy()


def _evaluate_openvino_chunk(arguments: tuple[int, int, int, int]) -> dict[str, Any]:
    start, count, seed, maximum_steps = arguments
    if _WORKER_CANDIDATE is None or _WORKER_BASELINE is None:
        raise RuntimeError("OpenVINO evaluation worker is not initialized")
    return evaluate_paired(
        _WORKER_CANDIDATE,
        _WORKER_BASELINE,
        deals=count,
        seed=seed + start,
        maximum_steps=maximum_steps,
        deal_offset=start,
    )


def evaluate_openvino_paired_parallel(
    model_path: str,
    policy_id: str,
    deals: int = 50_000,
    seed: int = 20260718,
    maximum_steps: int = 1000,
    workers: int = 12,
    progress_callback: Callable[[int, int], None] | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    if deals <= 0:
        raise ValueError("deals must be positive")
    worker_count = max(1, min(int(workers), int(deals), os.cpu_count() or 1))
    chunk_size = max(20, min(500, math.ceil(deals / (worker_count * 8))))
    chunks = [
        (start, min(chunk_size, deals - start), seed, maximum_steps)
        for start in range(0, deals, chunk_size)
    ]
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_key = {
        "model_path": str(Path(model_path).resolve()),
        "policy_id": policy_id,
        "deals": deals,
        "seed": seed,
        "maximum_steps": maximum_steps,
        "chunk_size": chunk_size,
    }
    checkpoint = {**checkpoint_key, "reports": {}}
    if checkpoint_file is not None and checkpoint_file.is_file():
        try:
            cached = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        if isinstance(cached, dict) and all(cached.get(key) == value for key, value in checkpoint_key.items()):
            checkpoint = cached
    cached_reports = checkpoint.setdefault("reports", {})
    reports = [report for report in cached_reports.values() if isinstance(report, dict)]
    completed_starts = {int(start) for start in cached_reports}
    pending_chunks = [chunk for chunk in chunks if chunk[0] not in completed_starts]
    completed_deals = sum(chunk[1] for chunk in chunks if chunk[0] in completed_starts)
    if progress_callback is not None:
        progress_callback(completed_deals, deals)
    if pending_chunks:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_openvino_worker,
            initargs=(str(model_path), policy_id),
        ) as executor:
            futures = {executor.submit(_evaluate_openvino_chunk, chunk): chunk for chunk in pending_chunks}
            for future in as_completed(futures):
                report = future.result()
                chunk = futures[future]
                reports.append(report)
                cached_reports[str(chunk[0])] = report
                completed_deals += chunk[1]
                if checkpoint_file is not None:
                    atomic_json(checkpoint_file, checkpoint)
                if progress_callback is not None:
                    progress_callback(completed_deals, deals)

    completed = sum(int(report["paired_seat_samples"]) for report in reports)
    delta_sum = sum(float(report["_delta_sum"]) for report in reports)
    delta_square_sum = sum(float(report["_delta_square_sum"]) for report in reports)
    candidate_wins = sum(int(report["_candidate_wins"]) for report in reports)
    mean_delta = delta_sum / completed if completed else 0.0
    if completed > 1:
        sample_variance = max(0.0, (delta_square_sum - delta_sum * delta_sum / completed) / (completed - 1))
        standard_error = math.sqrt(sample_variance / completed)
    else:
        standard_error = float("inf")
    margin = 1.96 * standard_error
    win_rate = candidate_wins / completed if completed else 0.0
    bounded = min(1.0 - 1e-6, max(1e-6, win_rate))
    failures = [failure for report in reports for failure in report.get("failures", [])]
    return {
        "requested_deals": deals,
        "paired_seat_samples": completed,
        "mean_reward_delta": mean_delta,
        "confidence_lower": mean_delta - margin,
        "confidence_upper": mean_delta + margin,
        "candidate_win_rate": win_rate,
        "elo_estimate": 400.0 * math.log10(bounded / (1.0 - bounded)),
        "illegal_actions": 0,
        "failed_samples": sum(int(report.get("failed_samples", 0)) for report in reports),
        "failures": failures[:100],
        "workers": worker_count,
        "chunks": len(chunks),
        "inference_backend": os.environ.get("CARD_AI_INFERENCE_BACKEND", "auto"),
    }
