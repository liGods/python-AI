from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ok_tasks.card_ai.continuous import ContinuousTrainer, ContinuousTrainingConfig
from ok_tasks.card_ai.evaluation import evaluate_openvino_paired_parallel
from ok_tasks.card_ai.heroes import HERO_REGISTRY, PASSIVE_OWNED_HEROES, SIMULATED_HEROES, iter_unverified_skills
from ok_tasks.card_ai.model_registry import ModelRegistry
from ok_tasks.card_ai.quality import RuntimeQualityGate, collect_runtime_quality
from ok_tasks.card_ai.real_data import convert_real_runs
from ok_tasks.card_ai.self_play import SelfPlayConfig, SelfPlayRunner
from ok_tasks.card_ai.sim2real import Sim2RealCalibrator
from ok_tasks.card_ai.training import train_three_seed_ensemble
from ok_tasks.card_ai.training_env import training_environment_report
from ok_tasks.card_ai.validation import run_resumable_property_validation


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _configure_utf8_console() -> None:
    """Keep PyTorch/ONNX Unicode progress output usable on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="百将牌AI模拟、训练、评测和部署工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    heroes = subparsers.add_parser("heroes", help="显示全英雄注册和验证状态")
    heroes.set_defaults(handler=_heroes)

    preflight = subparsers.add_parser("preflight", help="检查WSL2、CUDA和连续训练资源")
    preflight.add_argument("--project-root", default=".")
    preflight.set_defaults(handler=_preflight)

    simulate = subparsers.add_parser("simulate", help="生成确定性自我对战轨迹")
    simulate.add_argument("--games", type=int, default=1000)
    simulate.add_argument("--seed", type=int, default=20260718)
    simulate.add_argument("--workers", type=int, default=12)
    simulate.add_argument("--output", default="data/card_ai/training/manual_self_play")
    simulate.set_defaults(handler=_simulate)

    validate = subparsers.add_parser("validate", help="运行模拟器随机属性测试")
    validate.add_argument("--steps", type=int, default=10_000_000)
    validate.add_argument("--seed", type=int, default=20260718)
    validate.add_argument("--workers", type=int, default=12)
    validate.add_argument("--chunk-steps", type=int, default=5000)
    validate.add_argument("--checkpoint", default="data/card_ai/training/property_validation.json")
    validate.set_defaults(handler=_validate)

    quality = subparsers.add_parser("quality", help="检查真实自动化99.9%运行质量门禁")
    quality.add_argument("--runs", default="data/card_ai/runs")
    quality.set_defaults(handler=_quality)

    convert = subparsers.add_parser("convert-real", help="把真实牌局日志转换成v3训练轨迹")
    convert.add_argument("--runs", default="data/card_ai/runs")
    convert.add_argument("--output", default="data/card_ai/training/real_trajectories")
    convert.set_defaults(handler=_convert_real)

    sim2real = subparsers.add_parser("sim2real", help="分析真实手牌变化与基础模拟预测差异")
    sim2real.add_argument("--runs", default="data/card_ai/runs")
    sim2real.add_argument("--output", default="data/card_ai/training/sim2real_report.json")
    sim2real.set_defaults(handler=_sim2real)

    train = subparsers.add_parser("train", help="用v3轨迹训练三个随机种子的候选模型")
    train.add_argument("--trajectories", default="data/card_ai/training/**/trajectories/*.jsonl.gz")
    train.add_argument("--output", default="data/card_ai/training/manual_candidates")
    train.add_argument("--backbone", choices=("lstm", "transformer"), default="lstm")
    train.add_argument("--seed", type=int, default=20260718)
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("evaluate", help="固定种子配对评测OpenVINO候选")
    evaluate.add_argument("model")
    evaluate.add_argument("--deals", type=int, default=50_000)
    evaluate.add_argument("--seed", type=int, default=20260718)
    evaluate.add_argument("--workers", type=int, default=12)
    evaluate.set_defaults(handler=_evaluate)

    continuous = subparsers.add_parser("continuous", help="运行多日Sim2Real连续训练循环")
    continuous.add_argument("--project-root", default=".")
    continuous.add_argument("--target-games", type=int, default=2_000_000)
    continuous.add_argument("--batch-games", type=int, default=10_000)
    continuous.add_argument("--train-every", type=int, default=200_000)
    continuous.add_argument("--evaluation-deals", type=int, default=50_000)
    continuous.add_argument("--workers", type=int, default=12)
    continuous.set_defaults(handler=_continuous)

    registry = subparsers.add_parser("registry", help="检查或操作稳定模型注册表")
    registry.add_argument("action", choices=("status", "approve-canary", "promote", "rollback"))
    registry.add_argument("--root", default="data/card_ai/models")
    registry.add_argument("--version")
    registry.add_argument("--evaluation")
    registry.add_argument("--reason", default="手动回滚")
    registry.set_defaults(handler=_registry)
    return parser


def _heroes(_):
    _print(
        {
            "registered_heroes": len(HERO_REGISTRY),
            "simulated_heroes": list(SIMULATED_HEROES),
            "passive_curriculum_heroes": list(PASSIVE_OWNED_HEROES),
            "unverified_skills": [skill.to_dict() for skill in iter_unverified_skills()],
            "heroes": {hero: [skill.to_dict() for skill in skills] for hero, skills in HERO_REGISTRY.items()},
        }
    )


def _preflight(args):
    _print(training_environment_report(args.project_root))


def _simulate(args):
    _print(SelfPlayRunner().run_parallel(SelfPlayConfig(args.games, args.seed), args.output, args.workers))


def _validate(args):
    _print(
        run_resumable_property_validation(
            args.steps,
            args.seed,
            args.workers,
            checkpoint_path=args.checkpoint,
            chunk_steps=args.chunk_steps,
        )
    )


def _quality(args):
    metrics = collect_runtime_quality(args.runs)
    _print(RuntimeQualityGate().evaluate(metrics).to_dict())


def _convert_real(args):
    _print(convert_real_runs(args.runs, args.output))


def _sim2real(args):
    _print(Sim2RealCalibrator().analyze(args.runs, args.output))


def _train(args):
    paths = sorted(Path(".").glob(args.trajectories))
    _print(train_three_seed_ensemble(paths, args.output, args.backbone, args.seed))


def _evaluate(args):
    _print(
        evaluate_openvino_paired_parallel(
            args.model,
            "manual_candidate",
            deals=args.deals,
            seed=args.seed,
            workers=args.workers,
        )
    )


def _continuous(args):
    config = ContinuousTrainingConfig(
        target_games=args.target_games,
        batch_games=args.batch_games,
        train_every_games=args.train_every,
        evaluation_deals=args.evaluation_deals,
        workers=args.workers,
    )
    _print(ContinuousTrainer(args.project_root, config).run())


def _registry(args):
    registry = ModelRegistry(args.root)
    if args.action == "status":
        pointers = registry._pointers()
        manifests = {}
        for key, version in pointers.items():
            if key in {"stable", "candidate", "rollback"} and version:
                try:
                    manifests[key] = registry.load_manifest(version).to_dict()
                except OSError:
                    manifests[key] = {"version_id": version, "error": "manifest_missing"}
        _print({"pointers": pointers, "manifests": manifests})
        return
    if args.action == "rollback":
        _print({"stable": registry.rollback(args.reason)})
        return
    if not args.version or not args.evaluation:
        raise SystemExit("approve-canary/promote 必须提供 --version 和 --evaluation")
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    if args.action == "approve-canary":
        _print(registry.approve_canary(args.version, evaluation).to_dict())
        return
    quality = RuntimeQualityGate().evaluate(collect_runtime_quality("data/card_ai/runs"))
    _print(registry.promote(args.version, evaluation, quality).to_dict())


def main() -> None:
    _configure_utf8_console()
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
