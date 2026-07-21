from __future__ import annotations

import gc
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ok_tasks.card_ai.evaluation import evaluate_openvino_paired_parallel
from ok_tasks.card_ai.heroes import PASSIVE_OWNED_HEROES, SIMULATED_HEROES
from ok_tasks.card_ai.model_registry import ModelRegistry
from ok_tasks.card_ai.policies import OpenVinoPolicy, StableRulePolicy
from ok_tasks.card_ai.real_data import convert_real_runs
from ok_tasks.card_ai.self_play import SelfPlayConfig, SelfPlayRunner
from ok_tasks.card_ai.sim2real import Sim2RealCalibrator
from ok_tasks.card_ai.training import collect_training_samples, train_three_seed_ensemble
from ok_tasks.card_ai.trajectory import atomic_json


GIB = 1024**3


@dataclass(frozen=True)
class ContinuousTrainingConfig:
    target_games: int = 2_000_000
    batch_games: int = 10_000
    train_every_games: int = 200_000
    evaluation_deals: int = 50_000
    workers: int = 12
    maximum_steps: int = 1000
    replay_games: int = 50_000
    data_limit_gib: int = 350
    reserve_free_gib: int = 150
    seed: int = 20260718


class ContinuousTrainer:
    def __init__(self, project_root: str | Path, config: ContinuousTrainingConfig | None = None):
        self.project_root = Path(project_root)
        self.config = config or ContinuousTrainingConfig()
        self.data_root = self.project_root / "data" / "card_ai"
        self.training_root = self.data_root / "training"
        self.models_root = self.data_root / "models"
        self.state_path = self.training_root / "continuous_state.json"
        self.state = self._load_state()

    def _evaluation_worker_count(self) -> int:
        backend = os.environ.get("CARD_AI_INFERENCE_BACKEND", "auto").strip().lower()
        # Transformer CUDA sessions consume several GiB each on the RTX 4070.
        # Keep headroom for the coordinator and graphics driver during long evaluation.
        limit = 3 if backend == "cuda" else 4
        return min(limit, self.config.workers)

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {
                "completed_games": 0,
                "last_trained_games": 0,
                "cycle": 0,
                "evaluation_history": [],
                "status": "ready",
            }
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except ValueError:
            raise ValueError(f"连续训练状态文件损坏: {self.state_path}")
        return value

    def run(self) -> dict[str, Any]:
        self.state["status"] = "running"
        self._save_state()
        while int(self.state["completed_games"]) < self.config.target_games:
            disk_reason = self._disk_stop_reason()
            if disk_reason:
                self.state.update({"status": "paused_disk", "message": disk_reason})
                self._save_state()
                return self.state
            self.state["cycle"] = int(self.state.get("cycle", 0)) + 1
            cycle = int(self.state["cycle"])
            remaining = self.config.target_games - int(self.state["completed_games"])
            requested = min(self.config.batch_games, remaining)
            start_seed = self.config.seed + int(self.state["completed_games"])
            cycle_root = self.training_root / "self_play" / f"cycle_{cycle:05d}"
            calibration = Sim2RealCalibrator().analyze(
                self.data_root / "runs", self.training_root / "sim2real_report.json"
            )
            curriculum, hero_pool = self._curriculum(calibration)
            report = SelfPlayRunner().run_parallel(
                SelfPlayConfig(
                    requested,
                    start_seed,
                    self.config.maximum_steps,
                    hero_pool=hero_pool,
                ),
                cycle_root,
                self.config.workers,
            )
            report["curriculum"] = curriculum
            report["sim2real"] = calibration
            completed = int(report["completed_games"])
            self.state["completed_games"] = int(self.state["completed_games"]) + completed
            self.state["last_self_play"] = report
            if completed == 0:
                self.state.update({"status": "failed", "message": "本批次没有完成任何模拟牌局"})
                self._save_state()
                return self.state
            if int(self.state["completed_games"]) - int(self.state["last_trained_games"]) >= self.config.train_every_games:
                self.state["last_training"] = self._train_and_evaluate()
                self.state["last_trained_games"] = int(self.state["completed_games"])
            self._save_state()
            if self._plateau_reached():
                self.state.update({"status": "plateau", "message": "连续3次评测提升不足5 Elo"})
                self._save_state()
                return self.state
        self.state.update({"status": "completed", "message": "已达到目标模拟局数"})
        self._save_state()
        return self.state

    def _train_and_evaluate(self) -> dict[str, Any]:
        real_report = convert_real_runs(self.data_root / "runs", self.training_root / "real_trajectories")
        paths = self._mixed_replay_paths()
        if len(paths) < 10:
            return {"trained": False, "message": "完整轨迹不足10局", "real_data": real_report}
        architecture_results = []
        progress_path = self.training_root / "training_progress.json"
        for backbone_index, backbone in enumerate(("lstm", "transformer")):
            models_root = self.training_root / "candidates" / f"cycle_{self.state['cycle']:05d}" / backbone
            base_seed = self.config.seed + int(self.state["cycle"]) * 10
            required_names = ("training_metadata.json", "student.onnx", "student.xml", "student.bin")
            needs_training = any(
                not all((models_root / f"{backbone}_seed_{base_seed + offset}" / name).is_file() for name in required_names)
                for offset in range(3)
            )
            samples = None
            if needs_training:
                atomic_json(
                    progress_path,
                    {"schema_version": 1, "phase": "loading_samples", "backbone": backbone, "model_progress": 0.0},
                )
                samples = collect_training_samples(paths)
            result = train_three_seed_ensemble(
                paths,
                models_root,
                backbone=backbone,
                base_seed=base_seed,
                progress_path=progress_path,
                model_index_offset=backbone_index * 3,
                total_models=6,
                preloaded_samples=samples,
            )
            del samples
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            seed_results = []
            screening_deals = min(self.config.evaluation_deals, max(1_000, self.config.evaluation_deals // 10))
            for trained_seed in result["runs"]:
                seed_folder = Path(trained_seed["folder"])
                xml_path = seed_folder / "student.xml"
                if not xml_path.is_file():
                    seed_results.append({"seed": trained_seed["seed"], "error": "未导出OpenVINO模型"})
                    continue
                policy_id = f"candidate_{backbone}_{trained_seed['seed']}"
                model_index = backbone_index * 3 + len(seed_results) + 1
                atomic_json(
                    progress_path,
                    {
                        "schema_version": 1,
                        "phase": "screening",
                        "backbone": backbone,
                        "seed": trained_seed["seed"],
                        "model_index": model_index,
                        "total_models": 6,
                        "evaluation_deals": screening_deals,
                        "evaluation_completed": 0,
                    },
                )
                def publish_evaluation(completed_deals: int, total_deals: int) -> None:
                    atomic_json(
                        progress_path,
                        {
                            "schema_version": 1,
                            "phase": "screening",
                            "backbone": backbone,
                            "seed": trained_seed["seed"],
                            "model_index": model_index,
                            "total_models": 6,
                            "evaluation_deals": total_deals,
                            "evaluation_completed": completed_deals,
                            "evaluation_workers": self._evaluation_worker_count(),
                            "inference_backend": os.environ.get("CARD_AI_INFERENCE_BACKEND", "auto"),
                            "overall_progress": (model_index - 1 + completed_deals / total_deals) / 6,
                        },
                    )

                evaluation = evaluate_openvino_paired_parallel(
                    str(xml_path),
                    policy_id,
                    deals=screening_deals,
                    seed=self.config.seed + int(self.state["cycle"]) * self.config.evaluation_deals,
                    maximum_steps=self.config.maximum_steps,
                    workers=self._evaluation_worker_count(),
                    progress_callback=publish_evaluation,
                    checkpoint_path=self.training_root
                    / "evaluation_checkpoints"
                    / f"cycle_{int(self.state['cycle']):05d}_{backbone}_{trained_seed['seed']}_screening_{screening_deals}.json",
                )
                seed_results.append(
                    {"seed": trained_seed["seed"], "folder": str(seed_folder), "screening_evaluation": evaluation}
                )
            valid_seeds = [seed_result for seed_result in seed_results if seed_result.get("screening_evaluation")]
            if not valid_seeds:
                architecture_results.append(
                    {"backbone": backbone, "training": result, "seed_evaluations": seed_results}
                )
                continue
            best_seed = max(
                valid_seeds,
                key=lambda seed_result: seed_result["screening_evaluation"]["confidence_lower"],
            )
            trained_seed = next(
                run for run in result["runs"] if int(run["seed"]) == int(best_seed["seed"])
            )
            xml_path = Path(trained_seed["folder"]) / "student.xml"
            policy_id = f"candidate_{backbone}_{trained_seed['seed']}"
            model_index = backbone_index * 3 + result["runs"].index(trained_seed) + 1

            def publish_final_evaluation(completed_deals: int, total_deals: int) -> None:
                atomic_json(
                    progress_path,
                    {
                        "schema_version": 1,
                        "phase": "evaluation",
                        "backbone": backbone,
                        "seed": trained_seed["seed"],
                        "model_index": model_index,
                        "total_models": 6,
                        "evaluation_deals": total_deals,
                        "evaluation_completed": completed_deals,
                        "evaluation_workers": self._evaluation_worker_count(),
                        "inference_backend": os.environ.get("CARD_AI_INFERENCE_BACKEND", "auto"),
                        "overall_progress": (model_index - 1 + completed_deals / total_deals) / 6,
                    },
                )

            best_seed["evaluation"] = evaluate_openvino_paired_parallel(
                str(xml_path),
                policy_id,
                deals=self.config.evaluation_deals,
                seed=self.config.seed + int(self.state["cycle"]) * self.config.evaluation_deals,
                maximum_steps=self.config.maximum_steps,
                workers=self._evaluation_worker_count(),
                progress_callback=publish_final_evaluation,
                checkpoint_path=self.training_root
                / "evaluation_checkpoints"
                / f"cycle_{int(self.state['cycle']):05d}_{backbone}_{trained_seed['seed']}_evaluation_{self.config.evaluation_deals}.json",
            )
            architecture_results.append(
                {
                    "backbone": backbone,
                    "training": result,
                    "seed_evaluations": seed_results,
                    "evaluation": best_seed["evaluation"],
                    "folder": best_seed["folder"],
                    "seed": best_seed["seed"],
                }
            )
        valid = [result for result in architecture_results if result.get("evaluation")]
        if not valid:
            return {"trained": False, "message": "两个架构均未完成评测", "architectures": architecture_results}
        best = max(valid, key=lambda result: result["evaluation"]["confidence_lower"])
        folder = Path(best["folder"])
        version_base = f"cycle_{int(self.state['cycle']):05d}_{best['backbone']}_seed_{best['seed']}"
        version_id = version_base
        registry = ModelRegistry(self.models_root)
        collision_index = 1
        while (registry.versions / version_id).exists():
            version_id = f"{version_base}_r{collision_index}"
            collision_index += 1
        artifacts = {
            "openvino_xml": folder / "student.xml",
            "openvino_bin": folder / "student.bin",
            "onnx": folder / "student.onnx",
            "metadata": folder / "training_metadata.json",
        }
        chosen_training = next(
            run for run in best["training"]["runs"] if int(run["seed"]) == int(best["seed"])
        )
        manifest = registry.register_candidate(
            version_id,
            f"{best['backbone']}_distilled_action_ranker",
            "baijiangpai_observation_v3",
            artifacts,
            metrics={**chosen_training, "offline_evaluation": best["evaluation"]},
        )
        approved = False
        if best["evaluation"]["confidence_lower"] > 0 and best["evaluation"]["illegal_actions"] == 0:
            registry.approve_canary(version_id, best["evaluation"])
            approved = True
        history = self.state.setdefault("evaluation_history", [])
        history.append(
            {
                "cycle": self.state["cycle"],
                "version_id": version_id,
                "elo": best["evaluation"]["elo_estimate"],
                "confidence_lower": best["evaluation"]["confidence_lower"],
            }
        )
        return {
            "trained": True,
            "candidate": manifest.to_dict(),
            "approved_canary": approved,
            "real_data": real_report,
            "architectures": architecture_results,
        }

    def _stable_policy(self):
        registry = ModelRegistry(self.models_root)
        pointers = registry._pointers()
        stable = pointers.get("stable")
        if not stable:
            return StableRulePolicy()
        manifest = registry.load_manifest(stable)
        xml_name = manifest.files.get("openvino_xml")
        if not xml_name:
            return StableRulePolicy()
        return OpenVinoPolicy(str(registry.versions / stable / xml_name), stable)

    def _curriculum(self, calibration: dict[str, Any]) -> tuple[str, tuple[str | None, ...]]:
        completed = int(self.state.get("completed_games", 0))
        no_skill_until = min(200_000, max(1, int(self.config.target_games * 0.1)))
        if completed < no_skill_until:
            return "no_skill_doudizhu", (None,)
        if not calibration.get("passed"):
            return "no_skill_waiting_for_sim2real_99_5", (None,)
        passive_until = min(600_000, max(no_skill_until + 1, int(self.config.target_games * 0.3)))
        if completed < passive_until:
            return "verified_passive_skills", tuple(PASSIVE_OWNED_HEROES)
        return "verified_current_14", tuple(SIMULATED_HEROES)

    def _mixed_replay_paths(self) -> list[Path]:
        simulation = sorted(self.training_root.glob("self_play/cycle_*/trajectories/*.jsonl.gz"))
        real = sorted((self.training_root / "real_trajectories").glob("*.jsonl.gz"))
        maximum = self.config.replay_games
        recent_count = int(maximum * 0.7)
        historical_count = int(maximum * 0.2)
        real_count = maximum - recent_count - historical_count
        recent = simulation[-recent_count:]
        historical_pool = simulation[: max(0, len(simulation) - len(recent))]
        stride = max(1, len(historical_pool) // max(1, historical_count))
        historical = historical_pool[::stride][:historical_count]
        if real:
            repeated_real = [real[index % len(real)] for index in range(real_count)]
        else:
            repeated_real = []
        return [*recent, *historical, *repeated_real]

    def _disk_stop_reason(self) -> str | None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.data_root)
        if usage.free < self.config.reserve_free_gib * GIB:
            return f"磁盘剩余空间低于 {self.config.reserve_free_gib} GiB"
        size = sum(path.stat().st_size for path in self.data_root.rglob("*") if path.is_file())
        if size > self.config.data_limit_gib * GIB:
            return f"训练数据超过 {self.config.data_limit_gib} GiB 上限"
        return None

    def _plateau_reached(self) -> bool:
        history = self.state.get("evaluation_history", [])
        if len(history) < 4:
            return False
        gains = [history[index]["elo"] - history[index - 1]["elo"] for index in range(len(history) - 3, len(history))]
        return all(gain < 5.0 for gain in gains)

    def _save_state(self) -> None:
        atomic_json(self.state_path, {**self.state, "config": asdict(self.config)})
