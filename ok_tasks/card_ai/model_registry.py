from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ok_tasks.card_ai.quality import QualityGateResult
from ok_tasks.card_ai.trajectory import atomic_json


REGISTRY_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class ModelManifest:
    version_id: str
    model_type: str
    state_schema: str
    files: dict[str, str]
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "candidate"
    created_at: str = field(default_factory=_now)
    canary_games: int = 0
    canary_ratio: float = 0.1
    schema_version: int = REGISTRY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions = self.root / "versions"
        self.pointers_path = self.root / "pointers.json"
        self.runtime_stats_path = self.root / "runtime_stats.json"

    def _pointers(self) -> dict[str, Any]:
        if not self.pointers_path.is_file():
            return {"stable": None, "candidate": None, "rollback": None, "updated_at": _now()}
        try:
            value = json.loads(self.pointers_path.read_text(encoding="utf-8"))
        except ValueError:
            return {"stable": None, "candidate": None, "rollback": None, "updated_at": _now()}
        return value if isinstance(value, dict) else {"stable": None, "candidate": None, "rollback": None}

    def _runtime_stats(self) -> dict[str, Any]:
        if not self.runtime_stats_path.is_file():
            return {"policies": {}, "updated_at": _now()}
        try:
            value = json.loads(self.runtime_stats_path.read_text(encoding="utf-8"))
        except ValueError:
            return {"policies": {}, "updated_at": _now()}
        return value if isinstance(value, dict) else {"policies": {}, "updated_at": _now()}

    def register_candidate(
        self,
        version_id: str,
        model_type: str,
        state_schema: str,
        artifacts: dict[str, str | Path],
        metrics: dict[str, Any] | None = None,
    ) -> ModelManifest:
        version_folder = self.versions / version_id
        if version_folder.exists():
            raise FileExistsError(f"模型版本已经存在: {version_id}")
        version_folder.mkdir(parents=True)
        files = {}
        checksums = {}
        for name, source_value in artifacts.items():
            source = Path(source_value)
            if not source.is_file():
                raise FileNotFoundError(f"模型文件不存在: {source}")
            target = version_folder / source.name
            shutil.copy2(source, target)
            files[name] = target.name
            checksums[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = ModelManifest(version_id, model_type, state_schema, files, {**(metrics or {}), "sha256": checksums})
        atomic_json(version_folder / "manifest.json", manifest.to_dict())
        pointers = self._pointers()
        pointers.update({"candidate": version_id, "updated_at": _now()})
        atomic_json(self.pointers_path, pointers)
        return manifest

    def load_manifest(self, version_id: str) -> ModelManifest:
        value = json.loads((self.versions / version_id / "manifest.json").read_text(encoding="utf-8"))
        return ModelManifest(**value)

    def promote(
        self,
        version_id: str,
        offline_evaluation: dict[str, Any],
        quality_gate: QualityGateResult,
        minimum_canary_games: int = 200,
    ) -> ModelManifest:
        manifest = self.load_manifest(version_id)
        if not quality_gate.passed:
            raise ValueError("运行质量门禁未通过: " + "; ".join(quality_gate.reasons))
        if float(offline_evaluation.get("confidence_lower", 0.0)) <= 0.0:
            raise ValueError("离线配对评测的95%置信下界没有超过稳定模型")
        if int(offline_evaluation.get("illegal_actions", 0)) != 0:
            raise ValueError("候选模型产生过非法动作")
        if manifest.canary_games < minimum_canary_games:
            raise ValueError(f"真实小流量仅 {manifest.canary_games} 局，至少需要 {minimum_canary_games} 局")
        pointers = self._pointers()
        previous_stable = pointers.get("stable")
        baseline_id = previous_stable or "stable_rule_v3"
        candidate_real = self.real_performance(version_id)
        baseline_real = self.real_performance(baseline_id)
        if candidate_real["games"] < minimum_canary_games or baseline_real["games"] < minimum_canary_games:
            raise ValueError(
                f"真实对照样本不足: 候选 {candidate_real['games']} 局，稳定版本 {baseline_real['games']} 局"
            )
        if candidate_real["confidence_lower"] <= 0.0:
            raise ValueError("真实牌局胜率差的95%置信下界没有超过稳定版本")
        if candidate_real["submit_failure_rate"] > baseline_real["submit_failure_rate"]:
            raise ValueError("候选模型真实牌局提交失败率高于稳定版本")
        pointers.update(
            {"rollback": previous_stable, "stable": version_id, "candidate": None, "updated_at": _now()}
        )
        manifest.status = "stable"
        manifest.metrics["offline_evaluation"] = offline_evaluation
        manifest.metrics["quality_gate"] = quality_gate.to_dict()
        manifest.metrics["real_evaluation"] = {"candidate": candidate_real, "baseline": baseline_real}
        atomic_json(self.versions / version_id / "manifest.json", manifest.to_dict())
        atomic_json(self.pointers_path, pointers)
        return manifest

    def approve_canary(self, version_id: str, offline_evaluation: dict[str, Any]) -> ModelManifest:
        manifest = self.load_manifest(version_id)
        if float(offline_evaluation.get("confidence_lower", 0.0)) <= 0.0:
            raise ValueError("离线配对评测的95%置信下界没有超过稳定模型")
        if int(offline_evaluation.get("illegal_actions", 0)) != 0:
            raise ValueError("候选模型产生过非法动作")
        manifest.status = "canary"
        manifest.metrics["offline_evaluation"] = dict(offline_evaluation)
        atomic_json(self.versions / version_id / "manifest.json", manifest.to_dict())
        pointers = self._pointers()
        pointers.update({"candidate": version_id, "updated_at": _now()})
        atomic_json(self.pointers_path, pointers)
        return manifest

    def record_canary_game(self, version_id: str, won: bool, submit_failures: int = 0) -> ModelManifest:
        manifest = self.load_manifest(version_id)
        manifest.canary_games += 1
        canary = manifest.metrics.setdefault("canary", {"wins": 0, "losses": 0, "submit_failures": 0})
        canary["wins" if won else "losses"] += 1
        canary["submit_failures"] += int(submit_failures)
        atomic_json(self.versions / version_id / "manifest.json", manifest.to_dict())
        return manifest

    def record_runtime_game(self, policy_id: str, won: bool, submit_failures: int = 0) -> dict[str, Any]:
        stats = self._runtime_stats()
        policies = stats.setdefault("policies", {})
        policy = policies.setdefault(
            policy_id,
            {"games": 0, "wins": 0, "losses": 0, "submit_failures": 0, "recent": []},
        )
        policy["games"] = int(policy.get("games", 0)) + 1
        policy["wins"] = int(policy.get("wins", 0)) + int(bool(won))
        policy["losses"] = int(policy.get("losses", 0)) + int(not won)
        policy["submit_failures"] = int(policy.get("submit_failures", 0)) + int(submit_failures)
        recent = list(policy.get("recent", []))
        recent.append({"won": bool(won), "submit_failures": int(submit_failures)})
        policy["recent"] = recent[-500:]
        stats["updated_at"] = _now()
        atomic_json(self.runtime_stats_path, stats)
        return dict(policy)

    def real_performance(self, policy_id: str) -> dict[str, Any]:
        policy = self._runtime_stats().get("policies", {}).get(policy_id, {})
        games = int(policy.get("games", 0))
        wins = int(policy.get("wins", 0))
        rate = wins / max(1, games)
        pointers = self._pointers()
        baseline_id = pointers.get("stable") or "stable_rule_v3"
        baseline = self._runtime_stats().get("policies", {}).get(baseline_id, {})
        baseline_games = int(baseline.get("games", 0))
        baseline_rate = int(baseline.get("wins", 0)) / max(1, baseline_games)
        variance = rate * (1.0 - rate) / max(1, games)
        variance += baseline_rate * (1.0 - baseline_rate) / max(1, baseline_games)
        lower = rate - baseline_rate - 1.96 * variance**0.5
        return {
            "policy_id": policy_id,
            "games": games,
            "wins": wins,
            "win_rate": rate,
            "baseline_id": baseline_id,
            "baseline_games": baseline_games,
            "baseline_win_rate": baseline_rate,
            "confidence_lower": lower,
            "submit_failures": int(policy.get("submit_failures", 0)),
            "submit_failure_rate": int(policy.get("submit_failures", 0)) / max(1, games),
        }

    def reject_or_rollback(self, version_id: str, reason: str) -> str | None:
        pointers = self._pointers()
        if pointers.get("candidate") == version_id:
            manifest = self.load_manifest(version_id)
            manifest.status = "rejected"
            manifest.metrics["rejection_reason"] = reason
            atomic_json(self.versions / version_id / "manifest.json", manifest.to_dict())
            pointers.update({"candidate": None, "updated_at": _now(), "reason": reason})
            atomic_json(self.pointers_path, pointers)
            return pointers.get("stable")
        if pointers.get("stable") == version_id:
            return self.rollback(reason)
        return pointers.get("stable")

    def enforce_runtime_safety(self, policy_id: str) -> dict[str, Any] | None:
        pointers = self._pointers()
        if policy_id not in {pointers.get("stable"), pointers.get("candidate")}:
            return None
        policy = self._runtime_stats().get("policies", {}).get(policy_id, {})
        recent = list(policy.get("recent", []))
        if pointers.get("stable") == policy_id:
            if len(recent) >= 3 and all(int(item.get("submit_failures", 0)) > 0 for item in recent[-3:]):
                target = self.rollback("稳定模型连续3局发生提交失败")
                return {"action": "rollback", "target": target, "reason": "consecutive_submit_failures"}
            if len(recent) >= 40:
                current = recent[-20:]
                history = recent[:-20]
                current_rate = sum(bool(item.get("won")) for item in current) / len(current)
                history_rate = sum(bool(item.get("won")) for item in history) / len(history)
                if current_rate + 0.15 < history_rate:
                    target = self.rollback("稳定模型最近20局胜率明显退化")
                    return {"action": "rollback", "target": target, "reason": "recent_win_rate_regression"}
        if pointers.get("candidate") == policy_id and int(policy.get("games", 0)) >= 20:
            candidate_failure_rate = int(policy.get("submit_failures", 0)) / max(1, int(policy.get("games", 0)))
            baseline_id = pointers.get("stable") or "stable_rule_v3"
            baseline = self._runtime_stats().get("policies", {}).get(baseline_id, {})
            if int(baseline.get("games", 0)) >= 20:
                baseline_rate = int(baseline.get("submit_failures", 0)) / max(1, int(baseline.get("games", 0)))
                if candidate_failure_rate > baseline_rate:
                    target = self.reject_or_rollback(policy_id, "小流量提交失败率高于稳定版本")
                    return {"action": "reject", "target": target, "reason": "submit_failure_regression"}
        return None

    def rollback(self, reason: str) -> str | None:
        pointers = self._pointers()
        rollback_version = pointers.get("rollback")
        failed_stable = pointers.get("stable")
        pointers.update(
            {"stable": rollback_version, "rollback": failed_stable, "candidate": None, "updated_at": _now(), "reason": reason}
        )
        atomic_json(self.pointers_path, pointers)
        return rollback_version

    def select_for_game(self, game_id: str, allow_canary: bool, quality_gate: QualityGateResult) -> str | None:
        if not quality_gate.passed:
            return None
        pointers = self._pointers()
        candidate = pointers.get("candidate")
        if allow_canary and candidate:
            bucket = int.from_bytes(hashlib.sha256(game_id.encode()).digest()[:4], "big") % 1000
            manifest = self.load_manifest(candidate)
            if manifest.status == "canary" and bucket < int(manifest.canary_ratio * 1000):
                return candidate
        return pointers.get("stable")
