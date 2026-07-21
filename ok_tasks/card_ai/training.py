from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ok_tasks.card_ai.features import (
    HISTORY_FEATURE_SIZE,
    HISTORY_LIMIT,
    STATIC_FEATURE_SIZE,
    TEACHER_FEATURE_SIZE,
    encode_candidate,
    encode_teacher_hidden,
    opponent_targets,
    teammate_target,
)
from ok_tasks.card_ai.model_registry import ModelRegistry
from ok_tasks.card_ai.networks import build_student, build_teacher, require_torch
from ok_tasks.card_ai.trajectory import atomic_json, read_trajectory


@dataclass(frozen=True)
class TrainingConfig:
    backbone: str = "lstm"
    seed: int = 20260718
    epochs: int = 20
    teacher_epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 3e-4
    width: int = 384
    validation_ratio: float = 0.2


def _checkpoint_signature(config: TrainingConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backbone": config.backbone,
        "seed": config.seed,
        "width": config.width,
        "batch_size": config.batch_size,
        "teacher_epochs": config.teacher_epochs,
        "student_epochs": config.epochs,
    }


def _atomic_torch_save(value: dict[str, Any], path: str | Path) -> None:
    torch, _ = require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def _load_training_checkpoint(
    path: str | Path, config: TrainingConfig, device: str
) -> dict[str, Any] | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return None
    torch, _ = require_torch()
    try:
        value = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except (EOFError, OSError, RuntimeError, ValueError):
        return None
    signature = _checkpoint_signature(config)
    if not isinstance(value, dict) or any(value.get(key) != expected for key, expected in signature.items()):
        return None
    return value


def _rng_state(torch) -> dict[str, Any]:
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(torch, checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("python_rng") is not None:
        random.setstate(checkpoint["python_rng"])
    if checkpoint.get("numpy_rng") is not None:
        np.random.set_state(checkpoint["numpy_rng"])
    if checkpoint.get("torch_rng") is not None:
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def _write_training_progress(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    atomic_json(
        path,
        {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            **payload,
        },
    )


def collect_training_samples(paths: list[str | Path]) -> list[dict[str, Any]]:
    samples = []
    for path_value in paths:
        events = list(read_trajectory(path_value))
        if not events or not events[-1].get("terminal"):
            continue
        final_rewards = events[-1].get("rewards", {})
        for event in events:
            if event.get("event_type") != "decision":
                continue
            actor = event.get("actor")
            observation = event.get("observation")
            action = event.get("chosen_action")
            full_state = event.get("metadata", {}).get("full_state")
            if not actor or not isinstance(observation, dict) or not isinstance(action, dict):
                continue
            has_privileged = isinstance(full_state, dict)
            full_state = full_state if has_privileged else {}
            encoded = encode_candidate(observation, action)
            samples.append(
                {
                    **encoded,
                    "teacher_hidden": encode_teacher_hidden(full_state, actor),
                    "opponent_target": opponent_targets(full_state, actor),
                    "teammate_target": teammate_target(full_state, actor),
                    "return": float(final_rewards.get(actor, 0.0)),
                    "privileged": float(has_privileged),
                    "game_id": event.get("game_id"),
                }
            )
    return samples


def _stack(samples: list[dict[str, Any]], device: str):
    torch, _ = require_torch()
    return {
        "static": torch.as_tensor(np.stack([sample["static"] for sample in samples]), device=device),
        "history": torch.as_tensor(np.stack([sample["history"] for sample in samples]), device=device),
        "history_mask": torch.as_tensor(np.stack([sample["history_mask"] for sample in samples]), device=device),
        "role_index": torch.as_tensor([sample["role_index"] for sample in samples], device=device),
        "teacher_hidden": torch.as_tensor(np.stack([sample["teacher_hidden"] for sample in samples]), device=device),
        "opponent_target": torch.as_tensor(np.stack([sample["opponent_target"] for sample in samples]), device=device),
        "teammate_target": torch.as_tensor([sample["teammate_target"] for sample in samples], device=device),
        "return": torch.as_tensor([sample["return"] for sample in samples], device=device),
        "privileged": torch.as_tensor([sample["privileged"] for sample in samples], device=device),
    }


def _batches(samples: list[dict[str, Any]], batch_size: int, seed: int):
    indexes = list(range(len(samples)))
    random.Random(seed).shuffle(indexes)
    for start in range(0, len(indexes), batch_size):
        yield [samples[index] for index in indexes[start : start + batch_size]]


def train_candidate(
    trajectory_paths: list[str | Path],
    output_folder: str | Path,
    config: TrainingConfig,
    device: str | None = None,
    progress_path: str | Path | None = None,
    progress_context: dict[str, Any] | None = None,
    preloaded_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    torch, _ = require_torch()
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "training_checkpoint.pt"
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = str(target_device).startswith("cuda")
    if use_amp:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    context = dict(progress_context or {})
    _write_training_progress(
        progress_path,
        {**context, "phase": "loading_samples", "device": target_device, "model_progress": 0.0},
    )
    samples = preloaded_samples if preloaded_samples is not None else collect_training_samples(trajectory_paths)
    game_ids = sorted({sample["game_id"] for sample in samples})
    if len(game_ids) < 10:
        raise ValueError("至少需要10局完整v3轨迹才能训练候选模型")
    random.Random(config.seed).shuffle(game_ids)
    split = max(1, int(len(game_ids) * (1.0 - config.validation_ratio)))
    training_ids = set(game_ids[:split])
    training = [sample for sample in samples if sample["game_id"] in training_ids]
    validation = [sample for sample in samples if sample["game_id"] not in training_ids]
    if not validation:
        raise ValueError("验证集为空")
    batches_per_epoch = max(1, math.ceil(len(training) / config.batch_size))
    total_steps = (config.teacher_epochs + config.epochs) * batches_per_epoch
    checkpoint = _load_training_checkpoint(checkpoint_path, config, target_device)
    resumed_from_checkpoint = checkpoint is not None
    completed_teacher_epochs = min(
        config.teacher_epochs, max(0, int((checkpoint or {}).get("completed_teacher_epochs", 0)))
    )
    completed_student_epochs = min(
        config.epochs, max(0, int((checkpoint or {}).get("completed_student_epochs", 0)))
    )
    completed_steps = (completed_teacher_epochs + completed_student_epochs) * batches_per_epoch
    resumed_steps = completed_steps
    training_started = time.monotonic()

    def publish(phase: str, epoch: int, epochs: int, batch: int, loss_values: dict[str, float]) -> None:
        elapsed = max(0.001, time.monotonic() - training_started)
        progress = completed_steps / max(1, total_steps)
        run_steps = completed_steps - resumed_steps
        remaining = elapsed / max(1, run_steps) * max(0, total_steps - completed_steps)
        model_index = int(context.get("model_index", 1))
        total_models = int(context.get("total_models", 1))
        _write_training_progress(
            progress_path,
            {
                **context,
                "phase": phase,
                "device": target_device,
                "epoch": epoch,
                "epochs": epochs,
                "batch": batch,
                "batches": batches_per_epoch,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "model_progress": progress,
                "overall_progress": ((model_index - 1) + progress) / max(1, total_models),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(remaining, 1),
                "training_games": len(training_ids),
                "validation_games": len(game_ids) - len(training_ids),
                "training_samples": len(training),
                "validation_samples": len(validation),
                "losses": loss_values,
            },
        )

    teacher = build_teacher(config.width).to(target_device)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    if checkpoint is not None:
        teacher.load_state_dict(checkpoint["teacher_state"])
        if completed_teacher_epochs < config.teacher_epochs and checkpoint.get("teacher_optimizer"):
            teacher_optimizer.load_state_dict(checkpoint["teacher_optimizer"])
        if checkpoint.get("gradient_scaler"):
            gradient_scaler.load_state_dict(checkpoint["gradient_scaler"])
        _restore_rng_state(torch, checkpoint)
        if completed_teacher_epochs < config.teacher_epochs:
            checkpoint = None
            gc.collect()
            if use_amp:
                torch.cuda.empty_cache()

    def save_checkpoint(student=None, optimizer=None) -> None:
        _atomic_torch_save(
            {
                **_checkpoint_signature(config),
                "completed_teacher_epochs": completed_teacher_epochs,
                "completed_student_epochs": completed_student_epochs,
                "teacher_state": teacher.state_dict(),
                "teacher_optimizer": teacher_optimizer.state_dict(),
                "student_state": student.state_dict() if student is not None else None,
                "student_optimizer": optimizer.state_dict() if optimizer is not None else None,
                "gradient_scaler": gradient_scaler.state_dict(),
                **_rng_state(torch),
            },
            checkpoint_path,
        )

    for epoch in range(completed_teacher_epochs, config.teacher_epochs):
        teacher.train()
        for batch_index, batch_samples in enumerate(_batches(training, config.batch_size, config.seed + epoch), 1):
            batch = _stack(batch_samples, target_device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                prediction = teacher(batch["static"], batch["teacher_hidden"], batch["role_index"])
                per_sample = torch.nn.functional.mse_loss(prediction, batch["return"], reduction="none")
                loss = (per_sample * batch["privileged"]).sum() / batch["privileged"].sum().clamp(min=1.0)
            teacher_optimizer.zero_grad(set_to_none=True)
            gradient_scaler.scale(loss).backward()
            gradient_scaler.unscale_(teacher_optimizer)
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
            gradient_scaler.step(teacher_optimizer)
            gradient_scaler.update()
            completed_steps += 1
            if batch_index == batches_per_epoch or batch_index % max(1, batches_per_epoch // 10) == 0:
                publish("teacher", epoch + 1, config.teacher_epochs, batch_index, {"teacher_loss": float(loss.detach().cpu())})
        completed_teacher_epochs = epoch + 1
        save_checkpoint()
    student = build_student(config.backbone, config.width).to(target_device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    if checkpoint is not None and completed_student_epochs:
        if checkpoint.get("student_state") is None or checkpoint.get("student_optimizer") is None:
            completed_student_epochs = 0
            completed_steps = completed_teacher_epochs * batches_per_epoch
            resumed_steps = completed_steps
        else:
            student.load_state_dict(checkpoint["student_state"])
            optimizer.load_state_dict(checkpoint["student_optimizer"])
            _restore_rng_state(torch, checkpoint)
    checkpoint = None
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    teacher.eval()
    for epoch in range(completed_student_epochs, config.epochs):
        student.train()
        for batch_index, batch_samples in enumerate(
            _batches(training, config.batch_size, config.seed + 1000 + epoch), 1
        ):
            batch = _stack(batch_samples, target_device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    teacher_score = teacher(batch["static"], batch["teacher_hidden"], batch["role_index"])
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                score, opponent_prediction, teammate_prediction = student(
                    batch["static"], batch["history"], batch["history_mask"], batch["role_index"]
                )
                return_loss = torch.nn.functional.mse_loss(score, batch["return"])
                privilege = batch["privileged"]
                distillation_per_sample = torch.nn.functional.mse_loss(score, teacher_score, reduction="none")
                distillation_loss = (distillation_per_sample * privilege).sum() / privilege.sum().clamp(min=1.0)
                opponent_per_sample = torch.nn.functional.mse_loss(
                    opponent_prediction, batch["opponent_target"], reduction="none"
                ).mean(dim=(1, 2))
                opponent_loss = (opponent_per_sample * privilege).sum() / privilege.sum().clamp(min=1.0)
                teammate_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                    teammate_prediction, batch["teammate_target"], reduction="none"
                )
                teammate_loss = (teammate_per_sample * privilege).sum() / privilege.sum().clamp(min=1.0)
                loss = 0.65 * return_loss + 0.20 * distillation_loss + 0.10 * opponent_loss + 0.05 * teammate_loss
            optimizer.zero_grad(set_to_none=True)
            gradient_scaler.scale(loss).backward()
            gradient_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            completed_steps += 1
            if batch_index == batches_per_epoch or batch_index % max(1, batches_per_epoch // 10) == 0:
                publish(
                    "student",
                    epoch + 1,
                    config.epochs,
                    batch_index,
                    {
                        "total_loss": float(loss.detach().cpu()),
                        "return_loss": float(return_loss.detach().cpu()),
                        "distillation_loss": float(distillation_loss.detach().cpu()),
                        "opponent_loss": float(opponent_loss.detach().cpu()),
                        "teammate_loss": float(teammate_loss.detach().cpu()),
                    },
                )
        completed_student_epochs = epoch + 1
        save_checkpoint(student, optimizer)
    weights_path = output / "student.pt"
    teacher_path = output / "teacher.pt"
    torch.save(student.state_dict(), weights_path)
    torch.save(teacher.state_dict(), teacher_path)

    student.eval()
    validation_squared_error = 0.0
    opponent_squared_error = 0.0
    teammate_correct = 0.0
    validation_count = 0
    opponent_value_count = 0
    validation_batches = max(1, math.ceil(len(validation) / config.batch_size))
    with torch.no_grad():
        for validation_index, batch_samples in enumerate(
            _batches(validation, config.batch_size, config.seed + 2000), 1
        ):
            validation_batch = _stack(batch_samples, target_device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                validation_score, opponent_prediction, teammate_prediction = student(
                    validation_batch["static"],
                    validation_batch["history"],
                    validation_batch["history_mask"],
                    validation_batch["role_index"],
                )
            validation_squared_error += float(
                torch.nn.functional.mse_loss(
                    validation_score.float(), validation_batch["return"], reduction="sum"
                ).cpu()
            )
            opponent_squared_error += float(
                torch.nn.functional.mse_loss(
                    opponent_prediction.float(), validation_batch["opponent_target"], reduction="sum"
                ).cpu()
            )
            teammate_correct += float(
                (
                    (torch.sigmoid(teammate_prediction.float()) >= 0.5)
                    == validation_batch["teammate_target"].bool()
                )
                .float()
                .sum()
                .cpu()
            )
            validation_count += len(batch_samples)
            opponent_value_count += int(validation_batch["opponent_target"].numel())
            if validation_index == validation_batches or validation_index % max(1, validation_batches // 10) == 0:
                _write_training_progress(
                    progress_path,
                    {
                        **context,
                        "phase": "validation",
                        "device": target_device,
                        "batch": validation_index,
                        "batches": validation_batches,
                        "model_progress": 1.0,
                        "overall_progress": int(context.get("model_index", 1))
                        / max(1, int(context.get("total_models", 1))),
                    },
                )
    validation_mse = validation_squared_error / max(1, validation_count)
    opponent_mse = opponent_squared_error / max(1, opponent_value_count)
    teammate_accuracy = teammate_correct / max(1, validation_count)
    onnx_path = output / "student.onnx"
    dummy = (
        torch.zeros(2, STATIC_FEATURE_SIZE, device=target_device),
        torch.zeros(2, HISTORY_LIMIT, HISTORY_FEATURE_SIZE, device=target_device),
        torch.ones(2, HISTORY_LIMIT, dtype=torch.bool, device=target_device),
        torch.zeros(2, dtype=torch.long, device=target_device),
    )
    torch.onnx.export(
        student,
        dummy,
        onnx_path,
        input_names=["static", "history", "history_mask", "role_index"],
        output_names=["action_score", "opponent_cards", "teammate_ready"],
        dynamic_axes={name: {0: "batch"} for name in ("static", "history", "history_mask", "role_index", "action_score", "opponent_cards", "teammate_ready")},
        opset_version=18,
    )
    openvino_path = None
    try:
        import openvino as ov

        converted = ov.convert_model(onnx_path)
        openvino_path = output / "student.xml"
        ov.save_model(converted, openvino_path)
    except (ImportError, RuntimeError):
        openvino_path = None
    metadata = {
        "backbone": config.backbone,
        "seed": config.seed,
        "training_games": len(training_ids),
        "validation_games": len(game_ids) - len(training_ids),
        "training_samples": len(training),
        "validation_samples": len(validation),
        "validation_mse": validation_mse,
        "opponent_mse": opponent_mse,
        "teammate_accuracy": teammate_accuracy,
        "device": target_device,
        "state_schema": "baijiangpai_observation_v3",
        "onnx": str(onnx_path),
        "openvino": str(openvino_path) if openvino_path else None,
    }
    atomic_json(output / "training_metadata.json", metadata)
    _write_training_progress(
        progress_path,
        {
            **context,
            "phase": "model_complete",
            "device": target_device,
            "model_progress": 1.0,
            "overall_progress": int(context.get("model_index", 1))
            / max(1, int(context.get("total_models", 1))),
            "metrics": {
                "validation_mse": validation_mse,
                "opponent_mse": opponent_mse,
                "teammate_accuracy": teammate_accuracy,
            },
        },
    )
    checkpoint_path.unlink(missing_ok=True)
    checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp").unlink(missing_ok=True)
    return metadata


def train_three_seed_ensemble(
    trajectory_paths: list[str | Path],
    models_root: str | Path,
    backbone: str = "lstm",
    base_seed: int = 20260718,
    registry_root: str | Path | None = None,
    progress_path: str | Path | None = None,
    model_index_offset: int = 0,
    total_models: int = 3,
    preloaded_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(models_root)
    results = []
    for offset in range(3):
        seed = base_seed + offset
        folder = root / f"{backbone}_seed_{seed}"
        metadata_path = folder / "training_metadata.json"
        required_artifacts = (folder / "student.onnx", folder / "student.xml", folder / "student.bin")
        result = None
        if metadata_path.is_file() and all(path.is_file() for path in required_artifacts):
            try:
                cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cached = None
            if (
                isinstance(cached, dict)
                and cached.get("backbone") == backbone
                and int(cached.get("seed", -1)) == seed
            ):
                result = {**cached, "resumed": True}
                _write_training_progress(
                    progress_path,
                    {
                        "backbone": backbone,
                        "seed": seed,
                        "model_index": model_index_offset + offset + 1,
                        "total_models": total_models,
                        "phase": "model_reused",
                        "model_progress": 1.0,
                        "overall_progress": (model_index_offset + offset + 1) / max(1, total_models),
                    },
                )
        if result is None:
            result = train_candidate(
                trajectory_paths,
                folder,
                TrainingConfig(backbone=backbone, seed=seed),
                progress_path=progress_path,
                progress_context={
                    "backbone": backbone,
                    "seed": seed,
                    "model_index": model_index_offset + offset + 1,
                    "total_models": total_models,
                },
                preloaded_samples=preloaded_samples,
            )
        results.append({**result, "folder": str(folder)})
    best = min(results, key=lambda value: value["validation_mse"])
    version_id = f"{backbone}_{base_seed}_{len(trajectory_paths)}g"
    registry_result = None
    if registry_root is not None:
        folder = Path(best["folder"])
        artifacts = {"onnx": folder / "student.onnx", "metadata": folder / "training_metadata.json"}
        if (folder / "student.xml").is_file():
            artifacts.update({"openvino_xml": folder / "student.xml", "openvino_bin": folder / "student.bin"})
        manifest = ModelRegistry(registry_root).register_candidate(
            version_id,
            f"{backbone}_distilled_action_ranker",
            "baijiangpai_observation_v3",
            artifacts,
            metrics=best,
        )
        registry_result = manifest.to_dict()
    report = {"runs": results, "best": best, "registered_candidate": registry_result}
    atomic_json(root / f"ensemble_{backbone}_{base_seed}.json", report)
    return report
